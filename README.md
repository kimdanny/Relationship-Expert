# BotWise
## Building Conversational AI for Messenger Hackathon at Facebook 


1. **Idea**
    1. Appeal that it is an idea that everyone would love. Everyone has a situation that whether or not they should send some risky message or not. 
    
2. **Technology used**
    1. BERT(Bidirectional Transformer) Model
    2. Binary Sentiment Classification
    
3. **Scalability**
    1. Train and test with other dataset that has multiple sentiment
    2. Market demand
    
    
4. **Future improvement**
    - Train with other dataset that has multiple sentiment
    - Improve to implement a real chatbot


To get the model check point file, you can either  
- Run Sentiment_Analysis/Transformers for Sentiment Analysis.ipynb (Recommended to run it on GPU environment)
- Download [Bert-model.pt](https://drive.google.com/open?id=1w8S5IlQjexL2ERZAE_s1Q98jWA4LfhkD) and place it in your working directory, and set `path="./Bert-model.pt"` to fit `model_loader` function. 
