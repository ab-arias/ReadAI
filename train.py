from transformers import BigBirdTokenizer, BigBirdForSequenceClassification
# Additional imports for data preprocessing and training
import pandas as pd
import torch
import nltk
from nltk.corpus import cmudict
import re
import syllables

nltk.download('punkt')
nltk.download('cmudict')
cmu_dict = cmudict.dict()


tokenizer = BigBirdTokenizer.from_pretrained("google/bigbird-roberta-base")


# starting model (adjust accordingly)
model = BigBirdForSequenceClassification.from_pretrained("google/bigbird-roberta-base", num_labels=3)  # 3 classes for readability

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)


df_train = pd.read_csv('train.csv')
df_train.head()

# Define readability categories based on target values
def categorize_readability(target):
    if -3.7 <= target <= -1.5:
        return "College"
    elif -1.5 < target <= 0.75:
        return "High School"
    else:
        return "Elementary"
    
# Define readability categories based on target values
def categorize_FKR(target):
    if 0 <= target <= 30:
        return "College"
    elif 30 < target <= 60:
        return "High School"
    else:
        return "Elementary"
    

def flesch_kincaid_grade_level(excerpt):
    # Tokenize text into sentences and words
    para_text = excerpt.split('\n\n')
    meta_text = []
    for para in para_text:
        sents = nltk.sent_tokenize(para)
        meta_sents = []
        for s in sents:
            words = nltk.word_tokenize(s)
            meta_sents.extend(['<S>'] + words + ['</S>'])
        meta_text.extend(['<P>'] + meta_sents + ['</P>'])

    # Count words and syllables
    total_words = 0
    total_syl = 0
    total_one_syl = 0
    total_sents = meta_text.count('<S>')
    for word in meta_text:
        if not re.search(r'^\'[A-Za-z]+$|^[A-Za-z]+$', word):
            continue
        total_words += 1
        if word in cmu_dict.keys():
            curr_syls = 0
            for c in cmu_dict[word.lower()][0]:
                if c[len(c)-1].isnumeric():
                    total_syl += 1
                    curr_syls += 1
            if curr_syls == 1:
                total_one_syl += 1
        else:
            syls = syllables.estimate(word)
            total_syl += syls
            if syls == 1:
                total_one_syl += 1

    # Calculate Flesch-Kincaid grade level
    grade = 0.39 * (total_words / total_sents) + 11.8 * (total_syl / total_words) - 15.59
    return grade    
    

# Apply the categorization to the dataset
df_train['readability_category'] = df_train['target'].apply(categorize_readability)
df_train['flesch_kincaid_grade_level'] = df_train['excerpt'].apply(flesch_kincaid_grade_level)
df_train['fkr_category'] = df_train['flesch_kincaid_grade_level'].apply(categorize_FKR)
print(df_train.head())

label_mapping = {
    "Elementary": 0,
    "High School": 1,
    "College": 2
}

train_labels = ["Elementary", "High School", "College"]

# check if tokenizer works
text = df_train['excerpt'][0]
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

# Tokenize all
encoded_texts = [tokenizer(text, padding=True, truncation=True, return_tensors="pt") for text in df_train['excerpt']]

train_labels = df_train['readability_category']
train_labels = torch.tensor([label_mapping[label] for label in train_labels])  # Convert labels to numerical values





# # Define the loss function (CrossEntropyLoss for classification)
# loss_fn = torch.nn.CrossEntropyLoss()

# # Define the optimizer (e.g., Adam)
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

###############################################

# Training loop
# num_epochs = 4  # Adjust as needed
# model.train()
# for epoch in range(num_epochs):
#     total_loss = 0.0

#     # progress 0:600
#     for i in range(len(encoded_texts)):
#         input_ids = encoded_texts[i]['input_ids']
#         attention_mask = encoded_texts[i]['attention_mask']
#         label = train_labels[i]

#         optimizer.zero_grad()
#         outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=label.unsqueeze(0))
#         loss = outputs.loss
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#         print('epoch: ' + str(epoch + 1) + ' iteration: ' + str(i) + '/' + str(len(encoded_texts)))

#     average_loss = total_loss / len(encoded_texts)
#     print(f"Epoch {epoch + 1}, Average Loss: {average_loss:.4f}")

# Save the trained model (adjust name)
# model.save_pretrained("fine_tuned_bigbird_model_reading_formulas")