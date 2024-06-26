from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import scale, normalize
from sklearn.cluster import KMeans
from sklearn.decomposition import SparsePCA, PCA
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from IPython.display import display
from IPython.display import Markdown
from io import StringIO

import numpy as np
import pandas as pd
import requests
import json
import time
import openai
import google.generativeai as genai
import textwrap


### Functions ####
#define Jaccard Similarity function
def jaccard(list1, list2):
  intersection = len(list(set(list1).intersection(list2)))
  union = (len(list1) + len(list2)) - intersection
  return float(intersection) / union

# define speaker quality
def speaker_qual(speak_t, m=5, t=5): # m is the table size, t is the time per table
  target = t / m
  error = (speak_t - target)**2 + np.exp(3*(target - speak_t))
  th = t*m
  error[error>th] = th
  return (th-error) / th

#def customwrap(s,width=30):
#    return "<br>".join(textwrap.wrap(s,width=width))

def compute_dstats(metadata, d_parts, parts, 
                       d_pitches_vts_, id2pitchid, 
                       id2pitch, d_pitches_vts, d_table_part):
    title = metadata.topic_title
    N = d_parts.participant_id.unique().shape[0]
    n = parts.loc[parts.iteration_cycle==1, :].participant_id.unique().shape[0]
    participant_rank = d_pitches_vts_.groupby('created_participant_id').vote.sum().sort_values(ascending=False)

    winner_participant_id = participant_rank.index[0]
    winner_pitch_id = id2pitchid.loc[winner_participant_id, 'pitch_id']

    runnerup_participant_id = participant_rank.index[1]
    runnerup_pitch_id = id2pitchid.loc[runnerup_participant_id, 'pitch_id']

    supporters_w = d_pitches_vts_.groupby('pitch_id').voted_participant_id.unique()[winner_pitch_id]
    supporters_r = d_pitches_vts_.groupby('pitch_id').voted_participant_id.unique()[runnerup_pitch_id]

    w = id2pitch.loc[winner_participant_id, 'pitch_text']
    winner_idea = w
    s = supporters_w.shape[0]
    v = participant_rank.iloc[0]
    vperc = v / d_pitches_vts_.vote.sum()

    ntables = d_pitches_vts_.discussion_table_id.unique().shape[0]
    nrounds = d_pitches_vts.iteration_cycle.unique().shape[0]
    M = ntables * metadata.timePerTable
    m = nrounds * metadata.timePerTable
    polar = 1 - jaccard(d_pitches_vts_.loc[(d_pitches_vts_.iteration_cycle==4)].groupby('pitch_id').voted_participant_id.unique()[winner_pitch_id]\
                        , d_pitches_vts_.loc[(d_pitches_vts_.iteration_cycle==4)].groupby('pitch_id').voted_participant_id.unique()[runnerup_pitch_id])


    # For calculating engagement in each room - votes cast / votes available
    #ideas_in_room
    l = []

    # iterating through rooms
    for i,g in d_pitches_vts_.groupby('discussion_room_id'):
        nideas = g.pitch_id.unique().shape[0] # total number of ideas in the room
        vts_cast = g.vote.sum() # total number of votes cast in this room
        nusrs = g.voted_participant_id.unique().shape[0] # number of participants of this room
        if (g.iteration_cycle.unique()[0] == 1) & (nusrs>1): # when iteration cycle is 1
            e = vts_cast / (nusrs * (nusrs-1))
        else:
            e = vts_cast / (nusrs ** 2) # else
        if e > 1:
            e = 1
        l.append(e)

    eperc = np.nanmean(np.array(l))

    # list of participants in each round
    nusrs_round = [np.shape(parts.groupby('iteration_cycle').participant_id.unique()[i])[0] for i in np.array(range(nrounds)) + 1]

    d_table_part['st'] = d_table_part.speaking_time.copy()
    d_table_part.speaking_time = [int(i[:-2]) for i in d_table_part.st]
    avg_speaking_time = d_table_part.groupby(['participant_id']).apply(lambda x: x.speaking_time.mean()).mean()
    eperc_speaking = d_table_part.speaking_time.sum() / 60 / M

    dstats = pd.DataFrame(columns = ['turnout', 'ideas submitted', 'supporters of winning idea', 'rounds of discussion', 'total discussion time (min)', \
                                        'average speaking time (sec)'])
    dstats.loc[len(dstats.index)] = [metadata.totalParticipantsJoined, N, s, nrounds, M, avg_speaking_time]
    return (dstats, polar, eperc, 
            supporters_w, nrounds, winner_pitch_id, 
            title, N, n, w, s, v, vperc, M, m)

def compute_sankey(nrounds, d_pitches_vts):
    cols=list(np.array(range(nrounds))+1)
    df = pd.DataFrame(columns=cols)
    for p in d_pitches_vts.voted_participant_id.unique():
        tmp = d_pitches_vts.loc[(d_pitches_vts.voted_participant_id==p),['iteration_cycle', 'pitch_id', 'vote']]
        d = dict()
        (d[1], d[2], d[3], d[4]) = ([], [], [], [])
        for r in tmp.iterrows():
            d[r[1].iteration_cycle]+=list(np.repeat(r[1].pitch_id, r[1].vote))

        df = pd.concat([df, pd.DataFrame(dict([(k,pd.Series(v)) for k,v in d.items() ]))],
                ignore_index=True)
    df = pd.DataFrame(df.value_counts()).reset_index()
    df = df.rename({0:'value'}, axis='columns')
    df = df.rename(str, axis='columns')
    return df

def get_parent(d_table, df, idea_id):
  df1 = d_table.loc[d_table['pitch_id'] == idea_id] # get rows with idea id
  maxiter = max(df1['iteration_cycle']) # largest round achieved by idea
  df2 = df1.loc[df1['iteration_cycle'] == maxiter] # get rows with largest round achieved by idea
  roomid = df2['discussion_room_id'].unique()[0] # get room id of the idea's last room
  """if maxiter==d_table['iteration_cycle'].max(): 
      # if the idea is a finalist idea, then parent is the idea with the most votes
      parent = df.groupby(['pitch_id']).apply(lambda x: np.sum(x.vote)).idxmax()
  else: 
      r = df.loc[df['discussion_room_id'] == roomid] # get rows from d_pitches_vts with room id
      candidate_parents = r.pitch_id.unique()
      for i in candidate_parents: # loop across ideas that were in the room when the idea was defeated
          # if the idea is not a finalist idea, then parent is the idea that occurs in the next round
          if i in d_table.loc[d_table['iteration_cycle'] == maxiter+1, 'pitch_id'].unique():
              parent = i          
              """
  candidate_parents = df.loc[df['discussion_room_id'] == roomid, 'pitch_id'].unique()
  parent, votes = (None, 0)
  for i in candidate_parents:
      if df.loc[df['pitch_id'] == i, 'vote'].sum()>votes:
            parent = i
            votes = df.loc[df['pitch_id'] == i, 'vote'].sum()
  return (parent, roomid, maxiter)

def get_max_round(d_table, idea_id):
  df1 = d_table.loc[d_table['pitch_id'] == idea_id] # get rows with idea id
  maxiter = max(df1['iteration_cycle']) # largest round achieved by idea
  return maxiter

def compute_sunburst(d_pitches, nrounds, d_pitches_vts, d_rooms):
    df = d_pitches.drop(['user_id', 'pitch_audio', 'total_votes'], axis=1)
    df.columns = ['ideas in round 1', 'created_participant_id', 'display_name', 'pitch_text']
    df['ideas in round 2'] = np.nan
    df['ideas in round 3'] = np.nan
    df['ideas in round 4'] = np.nan
    df['votes'] = 0

    for i in range(1,nrounds):
        for p in df['ideas in round '+str(i)]:
            # find the room where idea p was discussed in round i
            roi = (d_pitches_vts.iteration_cycle==i) \
                            & (d_pitches_vts.pitch_id==p)
            if np.sum(roi)>0:
                room = d_pitches_vts.loc[roi, 'discussion_room_id'].unique()[0]
            else:
                continue

            # attach the winning idea for that room
            df.loc[df['ideas in round '+str(i)]==p, 'ideas in round '+str(i+1)] = d_rooms.loc[d_rooms.room_id==room, 'top_pitch'].values[0]

            # attach votes p received in round 1
            df.loc[df['ideas in round ' + str(i)]==p, 'votes'] = d_pitches_vts.loc[roi, 'vote'].sum()
    return df

def compute_idea_network(pitchid2pitchtext, d_pitches_vts):
    df = pd.DataFrame(columns=['From_ID', 'To_ID', 'Ideas', 'To_Name', 'Value', 'Votes', 'Round'])
    for n, i in enumerate(pitchid2pitchtext.index):
        for m, j in enumerate(pitchid2pitchtext.index[n:]):
            if i==j:
                continue

            i_supporters = d_pitches_vts.loc[d_pitches_vts.pitch_id==i, 'voted_participant_id'].unique()
            j_supporters = d_pitches_vts.loc[d_pitches_vts.pitch_id==j, 'voted_participant_id'].unique()

            if (len(i_supporters)==0) | (len(j_supporters)==0):
                continue

            # compute overlap of supporters using Jaccard
            sim = jaccard(i_supporters, j_supporters)

            if sim==0:
                continue
            # compute votes received by idea i
            votes = d_pitches_vts.loc[d_pitches_vts.pitch_id==i].vote.sum()

            # compute max round for idea i
            rnd = d_pitches_vts.loc[d_pitches_vts.pitch_id==i].iteration_cycle.max()

            # append to dataframe
            df.loc[len(df.index)] = [i, \
                                    j, \
                                    pitchid2pitchtext.loc[i].values[0], \
                                    pitchid2pitchtext.loc[j].values[0], \
                                    sim, \
                                    votes, \
                                    rnd]

    df.index = df.index.rename('Row ID')
    return df

def compute_idea_rank(matrix, pitchid2pitchtext):
    df = matrix.sum().sort_values(ascending=True).to_frame()
    df = df[df.index.notnull()]
    df.columns = ['vote']    
    df = df.assign(idea=[pitchid2pitchtext.loc[r[0]].iloc[0] for r in df.iterrows()])
    return df

def compute_polarisation_per_idea(pitchid2pitchtext, d_table_part, d_pitches_vts, demo):
    df = pd.DataFrame(columns=['Idea ID', 'Polarisation', 'Votes'])
    for i in pitchid2pitchtext.index:
        # find rooms where the idea was discussed
        i_rooms = d_table_part.loc[d_table_part.pitch_id==i, 'discussion_room_id'].unique()

        if len(i_rooms)==0:
            continue

        # participants who discussed the idea
        i_total = d_table_part.loc[d_table_part.discussion_room_id.isin(i_rooms), 'participant_id'].unique()

        # participants who voted for the idea
        i_supporters = d_pitches_vts.loc[d_pitches_vts.pitch_id==i, 'voted_participant_id'].unique()

        # democrats and republicans who discussed the idea
        dem_total = np.sum(demo.loc[demo.participant_id.isin(i_total), 'U.s. political affiliation']=='Democrat')
        rep_total = np.sum(demo.loc[demo.participant_id.isin(i_total), 'U.s. political affiliation']=='Republican')

        # democrats and republicans who voted for the idea
        dem_supporters = np.sum(demo.loc[demo.participant_id.isin(i_supporters), 'U.s. political affiliation']=='Democrat')
        rep_supporters = np.sum(demo.loc[demo.participant_id.isin(i_supporters), 'U.s. political affiliation']=='Republican')

        # probability that a democrat/republican vote for the idea
        if dem_total != 0:
            p_dem = dem_supporters / dem_total
        else:
            p_dem = np.nan
        if rep_total != 0:
            p_rep = rep_supporters / rep_total
        else:
            p_rep = np.nan

        # compute proportion
        if (p_rep + p_dem) != 0:
            pol = np.abs((p_dem / (p_rep + p_dem)) -0.5) * 2
        else:
            pol = np.nan

        # append to dataframe
        df.loc[len(df.index)] = [i, \
                                pol, \
                                d_pitches_vts.loc[d_pitches_vts.pitch_id==i, 'vote'].sum()]

    df.index = df.index.rename('row')
    df = df.assign(idea=[pitchid2pitchtext.loc[r[1]['Idea ID']]['pitch_text'] for r in df.iterrows()])
    return df

def impute_votes(id2pitch, d_pitches_vts_, winner_pitch_id):
    # Opinion matrix showing who voted (rows) on whose idea (cols)
    all_parts_ids = id2pitch.index.unique()
    idea_matrix = pd.pivot_table(d_pitches_vts_, values='vote',
                index='voted_participant_id',
                columns='pitch_id',
                dropna=False, aggfunc="sum")
    idea_matrix_ = idea_matrix.copy()
    
    # compute distance between ideas
    df = idea_matrix_.transpose().fillna(0) # idea id x users id

    from sklearn.neighbors import NearestNeighbors
    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(df.values)
    distances, indices = knn.kneighbors(df.values, n_neighbors=3)

    # get the index for winning idea
    index_for_ideas = df.index.tolist().index(winner_pitch_id)

    # find the indices for the similar ideas
    sim_ideas = indices[index_for_ideas].tolist()

    # distances between winner idea and similar ideas
    idea_distances = distances[index_for_ideas].tolist()

    # the position of winner idea in the list sim_ideas
    id_movie = sim_ideas.index(index_for_ideas)

    # remove winner idea from the list sim_ideas
    sim_ideas.remove(index_for_ideas)

    # remove winner idea from the list idea_distances
    idea_distances.pop(id_movie)

    """Predict votes on unseen ideas"""

    # copy df
    idea_matrix_imputed = df.copy()

    # find the nearest neighbors using NearestNeighbors(n_neighbors=3)
    number_neighbors = 3
    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(df.values)
    distances, indices = knn.kneighbors(df.values, n_neighbors=number_neighbors)

    for user_id in df.columns:
        # convert user_name to user_index
        user_index = df.columns.tolist().index(user_id)

        # t: idea_id, m: the row number of t in df
        for m,t in list(enumerate(df.index)):

            # find ideas without ratings by winner_participant_id
            if df.iloc[m, user_index] == 0:
                sim_ideas = indices[m].tolist()
                idea_distances = distances[m].tolist()

                # Generally, this is the case: indices[3] = [3 6 7]. The movie itself is in the first place.
                # In this case, we take off 3 from the list. Then, indices[3] == [6 7] to have the nearest NEIGHBORS in the list.
                if m in sim_ideas:
                    id_idea = sim_ideas.index(m)
                    sim_ideas.remove(m)
                    idea_distances.pop(id_idea)

                # However, if the percentage of ratings in the dataset is very low, there are too many 0s in the dataset.
                # Some movies have all 0 ratings and the movies with all 0s are considered the same movies by NearestNeighbors().
                # Then,even the movie itself cannot be included in the indices.
                # For example, indices[3] = [2 4 7] is possible if movie_2, movie_3, movie_4, and movie_7 have all 0s for their ratings.
                # In that case, we take off the farthest movie in the list. Therefore, 7 is taken off from the list, then indices[3] == [2 4].
                else:
                    sim_ideas = sim_ideas[:number_neighbors-1]
                    idea_distances = idea_distances[:number_neighbors-1]

            # movie_similarty = 1 - movie_distance
            idea_similarity = [1-x for x in idea_distances]
            idea_similarity_copy = idea_similarity.copy()
            nominator = 0

            # for each similar movie
            for s in range(0, len(idea_similarity)):

                # check if the rating of a similar movie is zero
                if df.iloc[sim_ideas[s], user_index] == 0:

                    # if the rating is zero, ignore the rating and the similarity in calculating the predicted rating
                    if len(idea_similarity_copy) == (number_neighbors - 1):
                        idea_similarity_copy.pop(s)

                    else:
                        idea_similarity_copy.pop(s-(len(idea_similarity)-len(idea_similarity_copy)))

                # if the rating is not zero, use the rating and similarity in the calculation
                else:
                    nominator = nominator + idea_similarity[s]*df.iloc[sim_ideas[s],user_index]

            # check if the number of the ratings with non-zero is positive
            if len(idea_similarity_copy) > 0:

                # check if the sum of the ratings of the similar movies is positive.
                if sum(idea_similarity_copy) > 0:
                    predicted_r = nominator/sum(idea_similarity_copy)

                # Even if there are some movies for which the ratings are positive, some movies have zero similarity even though they are selected as similar movies.
                # in this case, the predicted rating becomes zero as well
                else:
                    predicted_r = 0

            # if all the ratings of the similar movies are zero, then predicted rating should be zero
            else:
                predicted_r = 0

            # place the predicted rating into the copy of the original dataset
            idea_matrix_imputed.iloc[m,user_index] = predicted_r

    # melt and save
    idea_matrix_imputed = idea_matrix_imputed.rename_axis('idea_id')\
        .reset_index()\
        .melt('idea_id', value_name='votes', var_name='user_id')\
        .query('idea_id != user_id')\
        .reset_index(drop=True)
    # pivot table to ease some computations
    idea_matrix_imputed_pivoted = pd.pivot_table(
        idea_matrix_imputed, values='votes',
        index='user_id',
        columns='idea_id',
        dropna=False, aggfunc="sum")
    return (idea_matrix_imputed,idea_matrix_imputed_pivoted)

"""Participants map"""
def pca_participants(idea_matrix, d_pitches_vts, id2name):
    # PCA on opinion matrix
    # this creates a 2-d projection to cluster participants (based on similarity of
    # votes) or ideas (based on similarity of voters)
    pca = PCA(n_components=5, random_state=0)
    X = idea_matrix.fillna(0) # Participant ID X Idea ID
    X.columns = X.columns.astype(str)
    #pca.fit(normalize(X))
    components = pca.fit_transform(X)
    
    # calculate how many ideas were left in the last round
    #nfinal_ideas = d_pitches_vts_.loc[d_pitches_vts_.iteration_cycle == d_pitches_vts_.iteration_cycle.max(),'pitch_id'].unique().shape[0]
    # Now I calculate how many votes each idea got and use that to control the size of the idea dots
    #kmeans = KMeans(init="random", n_clusters=nfinal_ideas, n_init=4, random_state=0).fit(X)
    vts = list(np.zeros(len(X.index)))
    for i,p in enumerate(X.index): # loop across users
        roi = d_pitches_vts.loc[:,"created_participant_id"] == p # find rows with that user
        vts[i]=np.nansum(d_pitches_vts.loc[roi, "vote"].values) # sum votes user's idea received

    participants_map = pd.DataFrame(columns=['participant_id', 'x', 'y', 'votes'])
    participants_map.participant_id = idea_matrix.index
    participants_map.x = components[:,0]
    participants_map.y = components[:,1]
    participants_map.votes = vts
    participants_map = participants_map.loc[participants_map["participant_id"].notnull()]
    participants_map = participants_map.assign(
        participant=[id2name.loc[r[1]['participant_id']].iloc[0] for r in participants_map.iterrows()]
        )
    return participants_map

# sentence embedding (not using Jaseci)
def compute_idea_map(text, model="all-mpnet-base-v2"):
    transformer = SentenceTransformer(model)
    sentences = text.tolist() # Our sentences to encode
    embeddings = transformer.encode(sentences) # Sentences are encoded by calling model.encode()
    x = pd.DataFrame(embeddings)

    pca = PCA(n_components=2, random_state=0)
    x.columns = x.columns.astype(str)
    components = pca.fit_transform(x)
    df = pd.DataFrame(components, columns=['x', 'y'])
    df['texts'] = sentences
    df.texts = df.texts.str.wrap(30).apply(lambda x: x.replace('\n', '<br>'))
    return df

###JASECI_____________________________________________________________________
# check when the queued job has finished
def check_queued_job(queue_key):
  r = requests.get(url='http://psi.jaseci.org/js/walker_queue_check?task_id='+queue_key, headers={
        "Authorization": "Token a8db494387ba0dc65e6532053b18c9bb8c7e4ddc6bc93f6d398e228ab8d21066",
    })

  if r.json()['status']!='SUCCESS':
    while(r.json()['status']!='SUCCESS'):
      time.sleep(20)
      r = requests.get(url='http://psi.jaseci.org/js/walker_queue_check?task_id='+queue_key, headers={
            "Authorization": "Token a8db494387ba0dc65e6532053b18c9bb8c7e4ddc6bc93f6d398e228ab8d21066",
        })
  return r

def send_payload(pl):
  return requests.post(url='https://psi.jaseci.org/js/walker_run', json=pl, headers={
        "Authorization": "Token a8db494387ba0dc65e6532053b18c9bb8c7e4ddc6bc93f6d398e228ab8d21066",
  })

def import_discussion_data(d_pitches_vts, exp_name, title):
    """### Import discussion data"""

    df = d_pitches_vts.copy()
    df.iteration_cycle = df.iteration_cycle.astype('str')
    payload = {
        "name": "import_discussion_data",
        "ctx": {
            "discussion_info": [df.loc[r].fillna(value='null').to_dict() for r in d_pitches_vts.index],
            "id": exp_name,
            "topic": title
        }
    }

    r = send_payload(payload)
    return r

def delete_discussion(exp_name):
    """### delete discussion"""
    payload = {
        "name": "delete_discussion",
        "ctx": {
        "id":exp_name
    }
    }

    r = send_payload(payload)
    return r

def import_transcript_data(d_table_part, exp_name, title):
    df = d_table_part.copy()
    df = df.drop(['transcription_status','speaking_time', 'st'], axis=1)
    df.iteration_cycle = df.iteration_cycle.astype('str')
    d = [df.loc[r].fillna(value='null').to_dict() for r in df.index]
    payload = {
    "name": "import_transcript_data",
    "ctx": {
    "transcripts": d,
    "id": exp_name,
    "topic": title
    }
    }
    r = send_payload(payload)
    return r

def get_pitch_analytics(exp_name, model):
    if model == "K-means":
        plot_type="kmeans_pitch_graph_plot"
    elif model == "HDBScan":
        plot_type="hdbscan_pitch_graph_plot"

    # get_pitch_analytic: Allows you to get the pitch data with either kmeans or hdbscan data.
    payload = {
    "name": "get_pitch_analytics",
    "ctx": {
        "discussion_id":exp_name,
        "report_type": plot_type
    }
    }
    r = send_payload(payload)
    return r

def compute_idea_map_jaseci(d_pitches_vts, r):
    file = StringIO(r.text)
    df = pd.read_json(file)
    df = pd.DataFrame(df['report'][0])
    vts = []
    for p in df.texts: # loop across pitches
        roi = d_pitches_vts.loc[:,"pitch_text"] == p # find rows with that pitch
        vts.append(d_pitches_vts.loc[roi, "vote"].values.sum()) # sum total pitch votes
    df['votes'] = vts
    return df

def summarize_clusters(exp_name, cluster_method):
    if cluster_method=="HDBScan":
        method = "hdbscan_dialogue_cluster_summary"
    else:
        method = "kmeans_dialogue_cluster_summary"

    # gets stuck with async = True
    payload = {
        "name": "get_dialogue_analytics",
        "ctx": {
            "discussion_id": exp_name,
            "report_type": method
        }
    }
    r = send_payload(payload)
    return r

def summarize_tables(transcript):
    summaries = {}
    for i, g in transcript.groupby('table_id'):
        text = g.transcript.str.cat(sep=' ')
        payload = {
            "name": "summarize",
            "ctx": {
            "text": text,
            "reporting": True
            }
        }

        r = send_payload(payload)
        summaries[i] = json.loads(r.text)['report'][0]

    # save results
    df = pd.DataFrame(pd.Series(summaries), columns=['table ID'])
    df.index = df.index.rename('table ID')
    return df

def chat_gpt(prompt, messages, model="gpt-3.5-turbo-0125", client=None):
    # append user message
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    messages.append({"role": response.choices[0].message.role, "content": response.choices[0].message.content.strip()})
    return messages

### Gemini API
def chat_gemini(prompt, chat_session):
    prompt = prompt + " (remember, don't answer the question unless is relevant to the topic 'The most important questions in collective intelligence in 2024')"
    response = chat_session.send_message(prompt)

    try:
        answer = response.text
    except Exception as e:
        answer = f'{type(e).__name__}: {e}'

    return answer

def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

### STRIPE ____________________________________________________________________
