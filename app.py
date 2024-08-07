import streamlit as st
import plotly.graph_objects as go
from main import MachineLearningPipeline

ml_pipeline = MachineLearningPipeline()
cultures = ml_pipeline.cultures

st.title("CEPEA - PREDIÇÃO DE PREÇOS")

def select_box(cultures):
    culture = st.selectbox(
        label="Primeiro, selecione um indicador:",
        options=cultures.keys(),
        index=None,
        placeholder="Clique para selecionar um indicador...",
    )

    culture_id = st.selectbox(
        disabled=not culture,
        label="Agora, selecione uma variacao desse indicador:",
        options=cultures[culture].keys() if culture else [],
        format_func=lambda x: "{} - {}".format(x, cultures[culture][x]["title"]),
        index=None,
        placeholder="Clique para selecionar uma variacao...",
    )

    time_steps = st.selectbox(
        disabled=not culture_id,
        label="Escolha a quantidade de dias futuros a serem preditos:",
        options=[7, 15, 30],
        index=None,
        placeholder="Clique para selecionar a quantidade de dias",
    )
    return culture, culture_id, time_steps

def predict_figure(old_values, new_values):
    st.dataframe(old_values)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            y=old_values["value"].values,
            x=old_values["value"].index,
            line=dict(color="blue"),
            name="Historico",
            mode="lines+text",
            textposition="bottom left",
        )
    )
    fig.add_trace(
        go.Scatter(
            y=new_values,
            x=list(range(len(old_values) - 1, len(old_values) + time_steps)),
            line=dict(color="red"),
            name="Predito",
            mode="lines+text",
            textposition="bottom left",
        )
    )

    fig.update_layout(
        width=800,
        height=400,
        template="plotly_dark",
        title=dict(
            text=f"Predição {culture} - {time_steps} Dias".upper(),
            font=dict(
                size=24,
            ),
        ),
        xaxis_title=dict(text="Dias", font=dict(size=16)),
        yaxis_title=dict(text="Preço R$", font=dict(size=16)),
        xaxis=dict(tickfont=dict(size=14)),
        yaxis=dict(tickfont=dict(size=14)),
    )

    st.plotly_chart(fig)


culture, culture_id, time_steps = select_box(cultures)
conditions = (culture, culture_id, time_steps)

if st.button("Treinar Modelo") and all(conditions):
    ml_pipeline.time_steps = time_steps

    with st.status("Processando data") as status:
        st.write("Buscando o conjunto de dados...")
        st.write("Baixando o conjunto de dados...")
        st.write("Treinando modelo...")
        ml_pipeline.start_pipeline(culture_alias=culture, culture_id=culture_id)
        status.update(label="Processo Completo!", state="complete", expanded=False)

    data_length = time_steps * 2

    old_values = ml_pipeline.dataframe[["date", "value"]].tail(data_length).reset_index()
    new_values = ml_pipeline.predicted_values

    predict_figure(old_values, new_values)
