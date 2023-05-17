from models.sequential.GRU4Rec import GRU4Rec
import torch
import pandas as pd
import copy
import altair as alt

class GRU4Recold(GRU4Rec):
    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.topk_init = None
    
    def predict_scores(self,feed_dict,hidden_states: tuple = None):
        i_ids = torch.tensor([list(range(0,self.item_num))])
        history = feed_dict.squeeze(2).type(torch.int64)
        #import pdb;pdb.set_trace()
        if len(history.size()) == 1:
            #import pdb;pdb.set_trace()
            history = feed_dict.squeeze().unsqueeze(0)
            
        lengths = torch.tensor([len(x) for x in feed_dict.numpy()])
        batch_size = len(history)
       # history = feed_dict['history_items']  # [batch_size, history_max]
        #lengths = feed_dict['lengths']  # [batch_size]
        #
        
        #feed_dict['item_id'] = i_ids

        his_vectors = self.i_embeddings(history)

        # Sort and Pack
        sort_his_lengths, sort_idx = torch.topk(lengths, k=len(lengths))
        sort_his_vectors = his_vectors.index_select(dim=0, index=sort_idx)
        history_packed = torch.nn.utils.rnn.pack_padded_sequence(
            sort_his_vectors, sort_his_lengths.cpu(), batch_first=True)
        # RNN
       # output, hidden = self.rnn(history_packed, None)
        if hidden_states is None:
            output, hidden_states = self.rnn(history_packed)
        else:
            output, hidden_states = self.rnn(history_packed,hidden_states)
       # assert torch.equal(output[:,-1,:], hidden_states[-1, :, :])

        # Unsort
        unsort_idx = torch.topk(sort_idx, k=len(lengths), largest=False)[1]
        rnn_vector = hidden_states[-1].index_select(dim=0, index=unsort_idx)
        #import pdb;pdb.set_trace()
        # Predicts
        # pred_vectors = self.pred_embeddings(i_ids)
        pred_vectors = self.i_embeddings(i_ids)
        rnn_vector = self.out(rnn_vector)
        
        prediction = (rnn_vector[:, None, :] * pred_vectors).sum(-1)
        return prediction
    
    def forward(self, feed_dict,hidden_states: tuple = None):
        
        self.check_list = []
        #i_ids = feed_dict['item_id']  # [batch_size, -1]
        if self.topk_init is not None:
            i_ids = self.topk_init.unsqueeze(0)
        else:
            i_ids = torch.tensor([list(range(0,self.item_num))])
        
        history = feed_dict.squeeze(2).type(torch.int64)
        #import pdb;pdb.set_trace()
        if len(history.size()) == 1:
            #import pdb;pdb.set_trace()
            history = feed_dict.squeeze().unsqueeze(0)
            
        lengths = torch.tensor([len(x) for x in feed_dict.numpy()])
        batch_size = len(history)
       # history = feed_dict['history_items']  # [batch_size, history_max]
        #lengths = feed_dict['lengths']  # [batch_size]
        #
        
        #feed_dict['item_id'] = i_ids

        his_vectors = self.i_embeddings(history)

        # Sort and Pack
        sort_his_lengths, sort_idx = torch.topk(lengths, k=len(lengths))
        sort_his_vectors = his_vectors.index_select(dim=0, index=sort_idx)
        history_packed = torch.nn.utils.rnn.pack_padded_sequence(
            sort_his_vectors, sort_his_lengths.cpu(), batch_first=True)
        # RNN
       # output, hidden = self.rnn(history_packed, None)
        if hidden_states is None:
            output, hidden_states = self.rnn(history_packed)
        else:
            output, hidden_states = self.rnn(history_packed,hidden_states)
       # assert torch.equal(output[:,-1,:], hidden_states[-1, :, :])

        # Unsort
        unsort_idx = torch.topk(sort_idx, k=len(lengths), largest=False)[1]
        rnn_vector = hidden_states[-1].index_select(dim=0, index=unsort_idx)
        # Predicts
        pred_vectors = self.i_embeddings(i_ids)
        rnn_vector = self.out(rnn_vector)
        
        prediction = (rnn_vector[:, None, :] * pred_vectors).sum(-1)
        if self.topk_init is None:
            #import pdb;pdb.set_trace()
            pred = prediction
            sort_idx = (-pred).argsort(axis=1)[0]
            self.topk_init = sort_idx[:10]
        return prediction.view(batch_size, -1).mean(axis = 1).unsqueeze(0)/10,hidden_states
# model_check = GRU4Recold(args,corpus)
# model_check.load_state_dict(torch.load(file_path))

def plot_temp_coalition_pruning(df: pd.DataFrame,
                                pruned_idx: int,
                                plot_limit: int = 50,
                                solve_negatives: bool = False,
                                ):
    """Plots the coalition explainer process

    Parameters
    ----------
    df: pd.DataFrame
        Pruning algorithm output

    pruned_idx: int
        Index the explainer takes place

    plot_limit: int
        Window of events to show the explainer to
        Default is 50

    solve_negatives: bool
        Whether to remove negative importances of the background instances
    """

    def solve_negatives_method(df):
        
        negative_values = copy.deepcopy(df[df['Shapley Value'] < 0])
        for idx, row in negative_values.iterrows():
            corresponding_row = df[np.logical_and(df['t (event index)'] == row['t (event index)'], ~(df['Coalition'] == row['Coalition']))]
            df.at[corresponding_row.index, 'Shapley Value'] = corresponding_row['Shapley Value'].values[0] + row['Shapley Value']
            df.at[idx, 'Shapley Value'] = 0
        return df

    df = df[df['t (event index)'] >= -plot_limit
           ]
    if solve_negatives:
        df = solve_negatives_method(df)
    min_scale = int(df['Shapley Value'].min())
    max_scale = max(int(df['Shapley Value'].max()),1)
    base = (alt.Chart(df).encode(
        x=alt.X("t (event index)", axis=alt.Axis(title='t (event index)', labelFontSize=15,
                              titleFontSize=15)),
        y=alt.Y("Shapley Value",
                axis=alt.Axis(titleFontSize=15, grid=True, labelFontSize=15,
                              titleX=-28),
                scale=alt.Scale(domain=[min_scale, max_scale], )),
        color=alt.Color('Coalition', scale=alt.Scale(
            domain=['Sum of contribution of events \u2264 t'],
            range=["#618FE0"]), legend=alt.Legend(title=None, labelLimit=0,
                                                  fillColor="white",
                                                  labelFontSize=14,
                                                  symbolStrokeWidth=0,
                                                  symbolSize=50,
                                                  orient="top-left")),
    )
    )

    area_chart = base.mark_area(opacity=0.5)
    line_chart = base.mark_line()
    line = alt.Chart(pd.DataFrame({'x': [pruned_idx]})).mark_rule(
        color='#E17560').encode(x='x')

    text1 = alt.Chart(pd.DataFrame({'x': [pruned_idx - 2]})).mark_text(
        text='Pruning', angle=270, color='#E17560', fontSize=15,
        fontWeight='bold').encode(x='x')

    pruning_graph = (area_chart + line_chart + line + text1).properties(width=350,height=225)

    return pruning_graph