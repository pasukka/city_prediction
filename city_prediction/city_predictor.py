import pandas as pd
from rich.progress import Progress
from huggingface_hub import InferenceClient
import os

from city_prediction.config import load_config, Config


class CityPredictor:
    filepath: str
    messages_list: list
    model: str
    huggingface_hub_token: str

    def __init__(self, filepath: str):
        self.filepath = filepath

    def create_llm_prompt(self, prompt_template: str, text: str) -> str:
        message = (f'Текст: {text}\n')
        prompt = prompt_template
        prompt = prompt.replace('{message}', message)
        return prompt.strip()

    def get_city(self, prompt_template: str, message: str) -> str:
        prompt = self.create_llm_prompt(prompt_template, message)
        response = self.llm.text_generation(
            prompt, do_sample=False, max_new_tokens=20, stop_sequences=['.']).strip()
        response = response.replace("Ответ:", "")

        if (self.show_sentence):
            print(f'[SENTENCE]{message}\n')

        if (self.log_prompts):
            print(f'[INPUT PROMPT]: {prompt}\n')

        if self.log_llm_responses:
            print(f'[LLM RESPONSE]: {response}\n\n')

        return response

    def predict_cities(self) -> None:
        start = 0
        new_path = 'output_' + self.filepath
        if not os.path.exists(new_path):
            df = pd.DataFrame(columns=['message', 'city'])
            df.to_csv(new_path)
        else:
            start = len(pd.read_csv(new_path))

        self.llm = InferenceClient(model=self.model,
                                   timeout=15,
                                   token=self.huggingface_hub_token)
        with open(self.prompt_template_path, 'r', encoding='utf-8') as f:
            prompt_template = f.read().strip()

        total_sentences = len(self.message_list)
        with Progress() as progress:
            sentence_task = progress.add_task(
                description=f'[green]Sentence {start+1}/{total_sentences}',
                total=total_sentences
            )

            for ind in range(start, total_sentences):
                message = self.message_list[ind]
                city = self.get_city(prompt_template, message)
                progress.update(sentence_task,
                                advance=1,
                                description=f'[green]Sentence {ind+1}/{total_sentences}')

                df = pd.DataFrame({'message': [message],
                                  'city': [city]})
                df.index = [ind,]
                df.to_csv(new_path, mode='a', header=False)

    def __call__(self):
        config = load_config('config.yml')
        df = pd.read_csv(self.filepath)
        self.model = config.llm
        self.huggingface_hub_token = config.huggingface_hub_token
        
        self.message_list = df['message']
        self.log_prompts = config.log_prompts
        self.log_llm_responses = config.log_llm_responses
        self.prompt_template_path = config.prompt_template
        self.show_sentence = config.show_sentence

        self.predict_cities()
