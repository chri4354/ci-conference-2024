from shiny import App, reactive, ui
from shiny.types import ImgData
from models import *
from shinywidgets import output_widget, render_widget
from faicons import icon_svg as icon
from sklearn.cluster import KMeans
from dotenv import load_dotenv
from model_plots import *
from IPython.display import display
from IPython.display import Markdown
from pathlib import Path
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt

import shiny.experimental as x
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
import seaborn as sns
import plotly.graph_objects as go
import asyncio
import os 

# set environment
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sns.set_palette(["#8E59FF", "#E4D7FF", "#2C154D"])

### Configure API keys
load_dotenv()

# Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

### Load data#___________________________________________________________________________
question_list = [
    "Question 1",
    "Question 2",
    "Question 3",
]
# transcript data
transcript = pd.read_csv('Collective+intelligence+conference+2024.csv')
transcript = transcript.rename(columns={"Q1. Question 1 ( https://player.vimeo.com/external/363031257.hd.mp4?s=0c25528f8484030b0c45da6eda102836e11a58f8&amp;profile_id=175&amp;oauth2_token_id=57447761 )": "Q1",
                                             "Q2. Question 2 ( https://player.vimeo.com/external/363031257.hd.mp4?s=0c25528f8484030b0c45da6eda102836e11a58f8&amp;profile_id=175&amp;oauth2_token_id=57447761 )": "Q2",
                                             "Q3. Question 3 ( https://player.vimeo.com/external/363031257.hd.mp4?s=0c25528f8484030b0c45da6eda102836e11a58f8&amp;profile_id=175&amp;oauth2_token_id=57447761 )":"Q3"})
Q1_text = transcript.Q1.dropna().apply(lambda x: x.removeprefix("[Transcribed] "))
Q2_text = transcript.Q2.dropna().apply(lambda x: x.removeprefix("[Transcribed] "))
Q3_text = transcript.Q3.dropna().apply(lambda x: x.removeprefix("[Transcribed] "))

# compute idea map
idea_map1 = compute_idea_map(Q1_text)
idea_map2 = compute_idea_map(Q2_text)
idea_map3 = compute_idea_map(Q3_text)
transcript_text1 = "; ".join(Q1_text.to_list())
transcript_text2 = "; ".join(Q2_text.to_list())
transcript_text3 = "; ".join(Q3_text.to_list())
transcript_text = '**Responses to Q1:** ' + transcript_text1 + '. **Responses to Q2:** ' + transcript_text2 + '. **Responses to Q3:** ' + transcript_text3

# Compute Number of ideas submitted [TO DO]
N = transcript.shape[0]

gemini_model = genai.GenerativeModel(
  model_name="gemini-1.5-flash"
)

# ______________________________________________________________________
#### BUILD APPLICATION ### 
app_ui = ui.page_fluid(    
    ui.include_css("./css/my-styles.css"),   
    ui.head_content(ui.HTML(
        """
        <script src="https://cdn.jsdelivr.net/npm/gsap@3.12.5/dist/MotionPathPlugin.min.js"></script>
        <script src="https://unpkg.com/@dotlottie/player-component@latest/dist/dotlottie-player.mjs" type="module"></script>
        """),
        ),

    ui.layout_column_wrap(
        ui.value_box(
           "",
           "3 questions asked",
           "", 
           showcase=icon("clipboard-question"), 
           ), 
        ui.value_box(
           "",
           str(N)+" ideas submitted",
           "",
           showcase=icon("lightbulb")
        ), 
        ui.value_box(
           "", 
           "300+ attendees", 
           "", 
           showcase=icon("users"), 
           ), 
        style="margin: 20px auto; width: 90%;",
    ),
    
    ui.div(
        ui.p([
            "Created by: ",
            ui.a("psi.tech", target="_blank", href="https://psi.tech")
            ]),
        style="display: flex; justify-content: center; align-items: center; flex-wrap: wrap; margin:20px auto;"
        ),
    
    ui.div(
        ui.layout_column_wrap(
            ui.div(
                ui.h1("The collective intelligence community answers", class_="h1"),
                ), 
            ui.div(
                ui.HTML(
                    """<dotlottie-player src="https://lottie.host/2e532aa1-f609-4a99-95ff-fd12202b8c14/oDoZDcDCBB.json" background="transparent" speed="1" style="width: 300px; height: 300px;" loop autoplay></dotlottie-player>"""),
                style="display: flex; justify-content: center; align-items: center; " ,
                ), 
            style="margin: 20vmin auto; width: 90%;",
            ),
        ),
    
    ui.layout_column_wrap(
        ui.value_box(
           "",
           "Question 1",
           "How can AI and human intelligence be effectively integrated to enhance collective decision-making?", 
           showcase=icon("arrow-right-arrow-left"), 
           ), 
        ui.value_box(
           "",
           "Question 2",
           "What are the ethical implications of AI-assisted collective intelligence systems?",
           showcase=icon("arrow-down-up-across-line")
        ), 
        ui.value_box(
           "", 
           "Question 3", 
           "Under what conditions does collective intelligence emerge?", 
           showcase=icon("arrows-split-up-and-left"), 
           ), 
        style="margin: 20px auto; width: 90%;",
    ),
    
    ui.div(    
        ui.card(
            ui.card_header("Ask the AI what conference attendees said"),
            ui.layout_sidebar(
                ui.sidebar(
                    ui.input_action_button("chatbot_modal1", "How to interact with this card"),
                    ),
                ui.input_text_area("chat_input1", "", width='80%', placeholder="Ask a question about this survey..."),
                ui.tooltip(
                   ui.input_action_button("chat_submit1", "Submit", width='20%'), 
                   "Press the button only once. It takes a few seconds to generate a response.",
                   ),
                ),
            full_screen=True,
            max_height = "100vh",
            ),
        style="margin: 20vmin auto; width: 90%;",
        ),
    
    ui.card(
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_action_button("idea_map_modal1", "How to read this graph"),
                ui.input_slider(
                    "clusters_number1", label="Select a number of clusters",
                    min=2, max=5, value=2, step=1, ticks=True
                    ),
                ui.input_slider(
                    "clusters_number2", label="Select a number of clusters",
                    min=2, max=5, value=2, step=1, ticks=True
                    ),  
                ui.input_slider(
                    "clusters_number3", label="Select a number of clusters",
                    min=2, max=5, value=2, step=1, ticks=True
                    ), 
                ),
            ui.layout_column_wrap(
                ui.card(
                    ui.card_header("Q1. How can AI and human intelligence be effectively integrated to enhance collective decision-making?"),
                    ui.row(
                        output_widget("plot_idea_map1")
                        ),
                    full_screen=True,
                    ),
                
                ui.card(
                    ui.card_header("Q2. What are the ethical implications of AI-assisted collective intelligence systems?"),
                    ui.row(
                        output_widget("plot_idea_map2")
                        ),
                    full_screen=True,
                    ),

                ui.card(
                    ui.card_header("Q3. Under what conditions does collective intelligence emerge?"),
                    ui.row(
                        output_widget("plot_idea_map3")
                        ),
                    full_screen=True,
                    )
                ),
            )
        ),
)

### SERVER ###________________________________________________________________________
def server(input, output, session):    
    chat_session = gemini_model.start_chat(history=[])
    is_conversation_analysed = reactive.value(0)
    censored_tables = []
    
    @reactive.calc
    async def read_conversation():
        with ui.Progress(min=0, max=5) as p: # len(transcript.table_id.unique())
            p.set(message="Reading what people said.", 
                      detail="Hold tight...")
                
            try: 
                response = chat_session.send_message(
                    """A researcher is conducting a study. He asked the collective intelligence community 
                    'What are the 3 most important questions in collective intelligence in 2024?'. 
                    The questions were: 
                    1. How can AI and human intelligence be effectively integrated to enhance collective decision-making? 
                    2. What are the ethical implications of AI-assisted collective intelligence systems? 
                    3. Under what conditions does collective intelligence emerge?
                    He then asked those 3 questions to the attendees of the ACM Collective Intelligence Conference 2024. 
                    Read below the attendees' answers and answer only questions relevant to this topic: """ + \
                                                        " ".join(transcript_text))
                await asyncio.sleep(0.2)
                
            except Exception as e:
                    ui.notification_show("One idea has been BLOCKED.", type="warning")
                    print(f'{type(e).__name__}: {e}')

        is_conversation_analysed.set(1)        
        return

    # remove text input from text box
    @reactive.effect
    @reactive.event(input.chat_submit1)
    def _():
       ui.update_text("chat_input1", value="")

    # add text input to chat history
    @reactive.effect
    @reactive.event(input.chat_submit1)
    def _():
        ui.insert_ui(
            ui.div(
               ui.markdown(input.chat_input1()), 
               class_="chat-message-sent"
               ),
            selector="#chat_input1", 
            where="beforeBegin",
        ),

    @reactive.effect
    @reactive.event(input.chat_submit1)
    async def chat_output1():
        # if it is the first message, then read the conversation
        if len(chat_session.history) == 0:
            await read_conversation()

        # call Gemini API
        answer  = chat_gemini(input.chat_input1(), chat_session)
            
        ui.insert_ui(
            ui.div(
                ui.markdown(answer),
                class_="chat-message-bot"
                ),
            selector="#chat_input1", 
            where="beforeBegin",
        )
    
    @output
    @render_widget
    def plot_idea_map1():
        kmeans = KMeans(n_clusters=int(input.clusters_number1()), random_state=0, n_init="auto").fit(idea_map1[['x', 'y']])
        idea_map1['cluster_labels'] = kmeans.labels_
        fig = px.scatter(idea_map1, 
                        x="x", y="y",
                        hover_data="idea", 
                        color="cluster_labels", 
                        color_discrete_sequence=["#8E59FF","#E4D7FF", "#2C154D"],
                        opacity=.5, 
                    )
        fig.update_traces(marker=dict(line=dict(width=2, 
                                        color='black'))),
        fig.update_layout(
            xaxis_title="", yaxis_title="", 
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", 
            #xaxis=dict(scaleanchor='y', scaleratio=1)
        )
        fig.update_xaxes(showticklabels=False) # Hide x axis ticks 
        fig.update_yaxes(showticklabels=False) # Hide y axis ticks
        fig.update_coloraxes(showscale=False)
        return fig 

    @output
    @render_widget
    def plot_idea_map2():
        kmeans = KMeans(n_clusters=int(input.clusters_number2()), random_state=0, n_init="auto").fit(idea_map2[['x', 'y']])
        idea_map2['cluster_labels'] = kmeans.labels_
        fig = px.scatter(idea_map2, 
                        x="x", y="y",
                        hover_data="idea", 
                        color="cluster_labels", 
                        color_discrete_sequence=["#8E59FF","#E4D7FF", "#2C154D"],
                        opacity=.5, 
                    )
        fig.update_traces(marker=dict(line=dict(width=2, 
                                        color='black'))),
        fig.update_layout(
            xaxis_title="", yaxis_title="", 
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", 
            #xaxis=dict(scaleanchor='y', scaleratio=1)
        )
        fig.update_xaxes(showticklabels=False) # Hide x axis ticks 
        fig.update_yaxes(showticklabels=False) # Hide y axis ticks
        fig.update_coloraxes(showscale=False)
        return fig 
    
    @output
    @render_widget
    def plot_idea_map3():
        kmeans = KMeans(n_clusters=int(input.clusters_number3()), random_state=0, n_init="auto").fit(idea_map3[['x', 'y']])
        idea_map3['cluster_labels'] = kmeans.labels_
        fig = px.scatter(idea_map3, 
                        x="x", y="y",
                        hover_data="idea", 
                        color="cluster_labels", 
                        color_discrete_sequence=["#8E59FF","#E4D7FF", "#2C154D"],
                        opacity=.5, 
                    )
        fig.update_traces(marker=dict(line=dict(width=2, 
                                        color='black'))),
        fig.update_layout(
            xaxis_title="", yaxis_title="", 
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", 
            #xaxis=dict(scaleanchor='y', scaleratio=1)
        )
        fig.update_xaxes(showticklabels=False) # Hide x axis ticks 
        fig.update_yaxes(showticklabels=False) # Hide y axis ticks
        fig.update_coloraxes(showscale=False)
        return fig 
            


    ## ALL MODALS_______________________________________________________________________________________
    @reactive.effect
    @reactive.event(input.idea_map_modal1)
    def _():
        m = ui.modal(  
            ui.markdown("""
                        This scatter plot visually represents responders' ideas, with each dot representing an idea. Colors group similar ideas together. Ideas close to each other on the graph are more related. Use this graph to quickly understand thematic clusters (dots of the same color).
                        """),
            title="How to read this graph",  
            easy_close=True,  
        )  
        ui.modal_show(m)  

    @reactive.effect
    @reactive.event(input.chatbot_modal1)
    def _():
        m = ui.modal(
            ui.markdown(
                """
                Ask the AI about the answers provided by the conference attendees. Feel free to interact with the AI and explore what the community said in more detail.
                """
                ),  
            title="How to interact with the AI",  
            easy_close=False,  
            align="center"
        )  
        ui.modal_show(m)  


app_dir = Path(__file__).parent
print(app_dir)
# This is a shiny.App object. It must be named `app`.
app = App(app_ui, server=server, static_assets=app_dir / "www")