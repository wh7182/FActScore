from factscore.lm import LM
import openai
import sys
import time
import os
import numpy as np
import logging

class OpenAIModel(LM):

    def __init__(self, model_name, api_key, api_version, api_type, api_base, temp, cache_file=None):
        self.model_name = model_name
        self.api_key = api_key
        self.api_version = api_version
        self.api_type = api_type
        self.api_base = api_base
        self.temp = temp
        self.save_interval = 100
        super().__init__(cache_file)

    def load_model(self):
        openai.api_key = self.api_key
        openai.api_version = self.api_version
        openai.api_type = self.api_type
        openai.api_base = self.api_base
        self.model = self.model_name

    def _generate(self, prompt, max_sequence_length=2048, max_output_length=128):
        if self.add_n % self.save_interval == 0:
            self.save_cache()
        # return a tuple of string (generated text) and metadata (any format)
        # This should be about generating a response from the prompt, no matter what the application is
        response = run_chat_completion(prompt, max_tokens=max_sequence_length, temp=self.temp, engine=self.model_name)
        # Get the output from the response
        output = response["choices"][0]["message"]["content"]
        return output, response

def run_chat_completion(prompt, max_tokens=1000, temp=0.0, engine='gpt-4-32k'):

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    response = openai.ChatCompletion.create(
        engine=engine, 
        messages=messages, 
        max_tokens=max_tokens
    )

    return response
    
def call_ChatGPT(message, model_name="gpt-3.5-turbo", max_len=1024, temp=0.7, verbose=False):
    # call GPT-3 API until result is provided and then return it
    response = None
    received = False
    num_rate_errors = 0
    while not received:
        try:
            response = openai.ChatCompletion.create(model=model_name,
                                                    messages=message,
                                                    max_tokens=max_len,
                                                    temperature=temp)
            received = True
        except:
            # print(message)
            num_rate_errors += 1
            error = sys.exc_info()[0]
            if error == openai.error.InvalidRequestError:
                # something is wrong: e.g. prompt too long
                logging.critical(f"InvalidRequestError\nPrompt passed in:\n\n{message}\n\n")
                assert False
            
            logging.error("API error: %s (%d). Waiting %dsec" % (error, num_rate_errors, np.power(2, num_rate_errors)))
            time.sleep(np.power(2, num_rate_errors))
    return response


def call_GPT3(prompt, model_name="text-davinci-003", max_len=512, temp=0.7, num_log_probs=0, echo=False, verbose=False):
    # call GPT-3 API until result is provided and then return it
    response = None
    received = False
    num_rate_errors = 0
    while not received:
        try:
            response = openai.Completion.create(model=model_name,
                                                prompt=prompt,
                                                max_tokens=max_len,
                                                temperature=temp,
                                                logprobs=num_log_probs,
                                                echo=echo)
            received = True
        except:
            error = sys.exc_info()[0]
            num_rate_errors += 1
            if error == openai.error.InvalidRequestError:
                # something is wrong: e.g. prompt too long
                logging.critical(f"InvalidRequestError\nPrompt passed in:\n\n{prompt}\n\n")
                assert False
            logging.error("API error: %s (%d)" % (error, num_rate_errors))
            time.sleep(np.power(2, num_rate_errors))
    return response
