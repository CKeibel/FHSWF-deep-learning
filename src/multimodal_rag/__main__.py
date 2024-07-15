from multimodal_rag.frontend import gradio_frontend


def main() -> None:
    with gradio_frontend() as frontend:
        frontend.launch(share=True)


if __name__ == "__main__":
    main()
