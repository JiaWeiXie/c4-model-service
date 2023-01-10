from typing import List, Dict, Union, Tuple

import gradio as gr

from gradio.components import Component

from c4service.service import ModelService
from c4service.preprocess.py_lang import clear_source



class MainInterface:
    def __init__(
        self,
        service: ModelService,
    ) -> None:
        self.service = service

    def make_similarity_inputs(self) -> List[Component]:
        source_code = gr.Textbox(
            lines=5,
            max_lines=50,
            label="Source Code",
        )
        target_code = gr.Textbox(
            lines=5,
            max_lines=50,
            label="Target Code",
        )
        input_components = [source_code, target_code]
        return input_components

    def make_similarity_outputs(self) -> List[Component]:
        result_labels = gr.Label(label="Result")
        with gr.Row(equal_height=True):
            source_code_json = gr.JSON(label="Source Code Vector")
            target_code_json = gr.JSON(label="Target Code Vector")
        output_components = [result_labels, source_code_json, target_code_json]
        return output_components

    def similarity(
        self,
        source_code: str,
        target_code: str,
    ) -> Tuple[Dict[str, Union[str, float]], List[float], List[float]]:
        source_code = clear_source(source_code)
        target_code = clear_source(target_code)
        score, vec1, vec2 = self.service.predict(source_code, target_code, True)
        return {"Similarity": score}, vec1, vec2
    
    def vector(
        self,
        source_code: str,
    ) -> List[float]:
        source_code = clear_source(source_code)
        vec = self.service.predict_vector(source_code)
        return vec

    def render(self) -> None:
        gr.close_all()
        with gr.Blocks() as similarity_view:
            gr.Markdown("## C4 Model Code Similarity")
            with gr.Column(scale=1):
                inputs = self.make_similarity_inputs()
            
            sim_btn = gr.Button("Run")
            outputs = self.make_similarity_outputs() 
            sim_btn.click(
                self.similarity,
                inputs=inputs,
                outputs=outputs,
                api_name="similarity",
                scroll_to_output=True,
            )
            gr.Examples(
                [["print('Hello World.')", "print('Hello World.')"]],
                inputs,
                outputs,
                self.similarity,
                cache_examples=True,
            )
            
        with gr.Blocks() as vector_view:
            gr.Markdown("## C4 Model Code Vector")
            source_input = gr.Textbox(
                lines=5,
                max_lines=50,
                label="Source Code",
            )
            vec_btn = gr.Button("Run")
            source_output = gr.JSON(label="Source Code Vector")
            vec_btn.click(
                fn=self.vector,
                inputs=source_input,
                outputs=source_output,
                api_name="vector",
                scroll_to_output=True,
            )
            
        tab_views = gr.TabbedInterface(
            [similarity_view, vector_view],
            ["Code Similarity", "Code Vector"],
            title="C4 Model Service",
        )

        tab_views.launch(
            debug=True,
            show_error=True,
            server_name="0.0.0.0",
            server_port=8088,
        )