from langchain_google_genai import ChatGoogleGenerativeAI

from .interfaces import ModelLoader


class GeminiModelLoader(ModelLoader):

    def __init__(self, model_name: str):
        self.model_name = model_name

    def load(self):

        model = ChatGoogleGenerativeAI(
            model=self.model_name
        )

        return model