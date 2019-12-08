import torch
import torch.nn as nn

import os
from flask import Flask, request, make_response

app = Flask(__name__)

from json import dumps
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained('bert-base-uncased')


class BERTGRUSentiment(nn.Module):
    def __init__(self,
                 bert,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 bidirectional,
                 dropout):

        super().__init__()

        self.bert = bert

        embedding_dim = bert.config.to_dict()['hidden_size']

        self.rnn = nn.GRU(embedding_dim,
                          hidden_dim,
                          num_layers=n_layers,
                          bidirectional=bidirectional,
                          batch_first=True,
                          dropout=0 if n_layers < 2 else dropout)

        self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):

        # text = [batch size, sent len]

        with torch.no_grad():
            embedded = bert(text)[0]

        # embedded = [batch size, sent len, emb dim]

        _, hidden = self.rnn(embedded)

        # hidden = [n layers * n directions, batch size, emb dim]

        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])

        # hidden = [batch size, hid dim]

        output = self.out(hidden)

        # output = [batch size, out dim]

        return output


model = BERTGRUSentiment(bert,
                         hidden_dim=256,
                         output_dim=1,
                         n_layers=2,
                         bidirectional=True,
                         dropout=0.25)


def model_loader(path="./Bert-model.pt"):
    if os.path.isfile(path):
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)

        # Freeze all model parameters
        for param in model.parameters():
            param.requires_grad_(False)

        model.eval()
        return model

    else:
        print("No Path found")


max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']
init_token_idx = tokenizer.cls_token_id
eos_token = tokenizer.sep_token
eos_token_idx = tokenizer.convert_tokens_to_ids(eos_token)

model = model_loader()


def predict_sentiment(model, tokenizer, sentence):
    model.eval()
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:max_input_length - 2]
    indexed = [init_token_idx] + tokenizer.convert_tokens_to_ids(tokens) + [eos_token_idx]
    tensor = torch.LongTensor(indexed)
    tensor = tensor.unsqueeze(0)
    prediction = torch.sigmoid(model(tensor))
    return prediction.item()



@app.route("/get_advice",  methods=['POST'])
def postAdvice():
    print("server code has been called and request.json is about to shown")
    print(request.json)
    userText = request.json
    res = "Do not send it"
    output = predict_sentiment(model, tokenizer, userText)

    if output > 0.5: # happy
        res= "Feel free to send it"


    return res



if __name__ == "__main__":
    app.run()
