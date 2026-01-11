import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import cat, abs, tanh, sum, bmm, sigmoid
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoModel


class BertLSTM(nn.Module):
    def __init__(self, model_name='bert-base-uncased', hidden_dim=128):
        super(BertLSTM, self).__init__()

        self.bert = AutoModel.from_pretrained(model_name)
        for param in self.bert.parameters():
            param.requires_grad = False

        # bert_dim = 768
        bert_output_dim = 768
        lstm_input_dim = 256

        self.projection = nn.Sequential(
            nn.Linear(bert_output_dim, lstm_input_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2)
        )

        # self.lstm = nn.LSTM(bert_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.lstm = nn.LSTM(lstm_input_dim, hidden_dim, batch_first=True, bidirectional=True)

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

        # lstm_out_dim = hidden_dim * 2
        #
        #
        # self.attention = nn.Linear(lstm_out_dim, lstm_out_dim, bias=False)
        #
        #
        # combined_dim = lstm_out_dim * 4
        # self.fc = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(combined_dim, 64),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(64, 1)
        # )

    def encode_with_lstm(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            # embeddings: [Batch, Seq_Len, 768]
            embeddings = bert_out.last_hidden_state

        compressed_embeds = self.projection(embeddings)

        lstm_out, (hidden, cell) = self.lstm(compressed_embeds)

        final_vec = cat((hidden[-2], hidden[-1]), dim=1)

        return lstm_out, final_vec

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = bert_out.last_hidden_state

        compressed = self.projection(embeddings)

        _, (hidden, cell) = self.lstm(compressed)

        final_vec = torch.cat((hidden[-2], hidden[-1]), dim=1)

        raw = self.fc(final_vec).squeeze()
        return 4.0 * torch.sigmoid(raw) + 1.0

    # def forward(self, story_ids, story_mask, def_ids, def_mask):
    #     story_seq, _ = self.encode_with_lstm(story_ids, story_mask)
    #
    #     _, def_vec = self.encode_with_lstm(def_ids, def_mask)
    #     query = self.attention(def_vec).unsqueeze(2)  # [Batch, Hidden, 1]
    #     scores = bmm(story_seq, query).squeeze(2)  # [Batch, Seq_Len]
    #
    #     scores = scores.masked_fill(story_mask == 0, -1e9)
    #     attn_weights = F.softmax(scores, dim=1).unsqueeze(2)
    #
    #     context = sum(story_seq * attn_weights, dim=1)
    #
    #     u = context
    #     v = def_vec
    #     features = cat([u, v, abs(u - v), u * v], dim=1)
    #
    #     raw = self.fc(features).squeeze()
    #     return 4.0 * torch.sigmoid(raw) + 1.0


class LSTMRegressor(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128, pretrained_embeddings=None):
        super(LSTMRegressor, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = False

        # bidirectional=True allows the model to read context from both directions
        # batch_first=True ensures input format is (Batch, Seq_Len, Dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)

        self.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x shape: [batch_size, seq_len]

        embeds = self.embedding(x)

        lstm_out, (hidden, cell) = self.lstm(embeds)

        final_feature_vector = cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)

        prediction = self.fc(final_feature_vector)

        return prediction.squeeze()


class SiameseLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128, pretrained_embeddings=None):
        super(SiameseLSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = False  # Keep Frozen!
            # self.embedding.weight.requires_grad = True

        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.hidden_dim_total = hidden_dim * 2

        # self.attention = Attention(self.hidden_dim_total)
        # self.attention = DotAttention()
        self.attention = GeneralAttention(self.hidden_dim_total)

        combined_dim = self.hidden_dim_total * 4

        # self.fc = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(combined_dim, 128),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(128, 1)
        # )

        # self.fc = nn.Sequential(
        #     nn.Dropout(0.5),  # High dropout to fight overfitting
        #     nn.Linear(combined_dim, 128),
        #     nn.ReLU(),
        #     nn.Dropout(0.4),  # Second dropout
        #     nn.Linear(128, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 1)
        # )

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def encode(self, x, lengths):
        embeds = self.embedding(x)

        packed_embeds = pack_padded_sequence(embeds, lengths.cpu(), batch_first=True, enforce_sorted=False)

        packed_output, (hidden, cell) = self.lstm(packed_embeds)

        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # final_hidden = cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        final_hidden = cat((hidden[-2], hidden[-1]), dim=1)

        return output, final_hidden

    def forward(self, story_seq, story_len, def_seq, def_len):
        # story_outputs, _ = self.encode(story_seq, story_len)
        # _, def_final_vec = self.encode(def_seq, def_len)
        #
        # story_context, attn_weights = self.attention(story_outputs)

        story_keys, _ = self.encode(story_seq, story_len)
        _, def_query = self.encode(def_seq, def_len)

        context, _ = self.attention(def_query, story_keys, story_keys)

        u = context
        v = def_query
        # u = story_context
        # v = def_final_vec

        features = cat([
            u,
            v,
            abs(u - v),
            u * v
        ], dim=1)

        # return self.fc(features).squeeze()

        raw_score = self.fc(features).squeeze()

        return 4.0 * sigmoid(raw_score) + 1.0


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, encoder_outputs):
        energy = tanh(self.attn(encoder_outputs))  # [Batch, Seq_Len, Hidden]

        scores = self.v(energy)  # [Batch, Seq_Len, 1]

        weights = F.softmax(scores, dim=1)  # [Batch, Seq_Len, 1]

        context_vector = sum(weights * encoder_outputs, dim=1)  # [Batch, Hidden]

        return context_vector, weights


class DotAttention(nn.Module):
    def __init__(self):
        super(DotAttention, self).__init__()

    def forward(self, query, keys, values):
        query = query.unsqueeze(1)

        scores = bmm(query, keys.transpose(1, 2))

        weights = F.softmax(scores, dim=2)  # [Batch, 1, Seq_Len]

        context = bmm(weights, values)

        return context.squeeze(1), weights.squeeze(1)


class GeneralAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(GeneralAttention, self).__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, query, keys, values):
        query_proj = self.W(query)

        query_proj = query_proj.unsqueeze(1)

        scores = bmm(query_proj, keys.transpose(1, 2))

        weights = F.softmax(scores, dim=2)

        context = bmm(weights, values)

        return context.squeeze(1), weights.squeeze(1)
