import gradio as gr
from contextlib import contextmanager


class SettingsService:
    settings = {
        "generation_config": {
            "temperature": 1.0,
            "max_new_tokens": 250,
            "pad_token_id": "<PAD>",  # TODO: set the pad token id
            "no_repeat_ngram_size": 3,
        }
    }

    @staticmethod
    def change_model(model_choice):
        pass

    @staticmethod
    def update_generation_config():
        pass


@contextmanager
def settings_tab():
    with gr.Tab("Settings") as settings_tab:
        gr.Markdown("# Model selection")
        with gr.Row():
            model_choice = gr.Dropdown(
                choices=[],  # Todo: set the choices (models)
                value=[],  # Todo: set the default value
                label="Model",
            )
            btn = gr.Button("Select")
            btn.click(SettingsService.change_model, inputs=model_choice, outputs=None)
        gr.Markdown("# Generation config")
        with gr.Column():
            new_tokens_slider = gr.Slider(
                minimum=50,
                maximum=1000,
                step=1.0,
                label="Max. new tokens",
                info="Test info",
                value=SettingsService.settings["generation_config"]["max_new_tokens"],
            )
            temp_slider = gr.Slider(
                minimum=0,
                maximum=99,
                step=1.0,
                label="Temperature",
                value=SettingsService.settings["generation_config"]["temperature"],
            )
            n_gram_slider = gr.Slider(
                minimum=0,
                maximum=15,
                step=1.0,
                label="No repeat n-gram size",
                value=SettingsService.settings["generation_config"][
                    "no_repeat_ngram_size"
                ],
            )
            btn = gr.Button("Save")
            btn.click(
                SettingsService.update_generation_config,
                inputs=[new_tokens_slider, temp_slider, n_gram_slider],
                outputs=None,
            )
    yield settings_tab
