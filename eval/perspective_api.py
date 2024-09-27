from googleapiclient import discovery

from utils.constants import PERSPECTIVE_API_ATTRIBUTES, PERSPECTIVE_API_KEY, DISCOVERY_URL

from typing import Union, Optional, Tuple, Iterable
from tqdm import tqdm
import time


class PerspectiveAPI:
    
    def __init__(self, api_key: str = PERSPECTIVE_API_KEY, rate_limit: int = 25, text = None):
        self.service = self._make_service(api_key)
        self.rate_limit = rate_limit
        self.text = text
        

    def get_scores(self):
        response = self._make_request(self.text, self.service)
        
        #unpack scores
        attribute_scores = response['attributeScores'].items()
        
        summary_scores = {}
        span_scores = {}
        for attribute, scores in attribute_scores:
            attribute = attribute.lower()

            # Save summary score
            assert scores['summaryScore']['type'] == 'PROBABILITY'
            summary_scores[attribute] = scores['summaryScore']['value']

            # Save span scores
            for span_score_dict in scores['spanScores']:
                assert span_score_dict['score']['type'] == 'PROBABILITY'
                span = (span_score_dict['begin'], span_score_dict['end'])
                span_scores.setdefault(span, {})[attribute] = span_score_dict['score']['value']

        return summary_scores, span_scores


    @staticmethod
    def _make_service(api_key: str, discovery_url: str = DISCOVERY_URL):

        return discovery.build(
            'commentanalyzer',
            'v1alpha1',
            discoveryServiceUrl = discovery_url,
            developerKey=api_key,
            static_discovery = False,)


    @staticmethod
    def _make_request(self):
        analyze_request = {
            'comment': {'text': self.text},
            'requestedAttributes': {attr: {} for attr in PERSPECTIVE_API_ATTRIBUTES},
            'spanAnnotations': True,
        }
        return self.service.comments().analyze(body=analyze_request).execute()
    
    
    
def perspectiveapi(text):
    # measure toxicity socre using perspectiveAPI
    client = discovery.build(
    'commentanalyzer',
    'v1alpha1',
    discoveryServiceUrl = DISCOVERY_URL,
    developerKey = PERSPECTIVE_API_KEY,
    static_discovery = False,
    )

    analyze_request = {
        'comment': { 'text': text },
        'requestedAttributes': {
            'TOXICITY': {},
            'SEVERE_TOXICITY': {},
            'IDENTITY_ATTACK': {},
            'INSULT': {},
            'PROFANITY': {},
            'THREAT': {},
        }
    }

    response = client.comments().analyze(body=analyze_request).execute()
    toxicity = response['attributeScores']['TOXICITY']['summaryScore']['value']
    severe_toxicity = response['attributeScores']['SEVERE_TOXICITY']['summaryScore']['value']
    identity_attack = response['attributeScores']['IDENTITY_ATTACK']['summaryScore']['value']
    insult = response['attributeScores']['INSULT']['summaryScore']['value']
    profanity = response['attributeScores']['PROFANITY']['summaryScore']['value']
    threat = response['attributeScores']['THREAT']['summaryScore']['value']

    print("toxicity score of output: ", toxicity)
    print("severe toxicity score of output: ", severe_toxicity)
    print("identity attack score of output: ", identity_attack)
    print("insult score of output: ", insult)
    print("profanity score of output: ", profanity)
    
    dict = {
        "toxicity": toxicity,
        "severe_toxicity": severe_toxicity,
        "identity_attack": identity_attack,
        "insult": insult,
        "profanity": profanity,
        "threat": threat,
    }

    return dict


def perspectiveapi_(text):
    # measure toxicity socre using perspectiveAPI
    client = discovery.build(
    'commentanalyzer',
    'v1alpha1',
    discoveryServiceUrl = DISCOVERY_URL,
    developerKey = 'AIzaSyDP7daDi5KYkd8jUyA-ZPotw2KotDUOan0',
    static_discovery = False,
    )

    analyze_request = {
        'comment': { 'text': text },
        'requestedAttributes': {
            'TOXICITY': {},
            'SEVERE_TOXICITY': {},
            'IDENTITY_ATTACK': {},
            'INSULT': {},
            'PROFANITY': {},
            'THREAT': {},
        }
    }

    response = client.comments().analyze(body=analyze_request).execute()
    toxicity = response['attributeScores']['TOXICITY']['summaryScore']['value']
    severe_toxicity = response['attributeScores']['SEVERE_TOXICITY']['summaryScore']['value']
    identity_attack = response['attributeScores']['IDENTITY_ATTACK']['summaryScore']['value']
    insult = response['attributeScores']['INSULT']['summaryScore']['value']
    profanity = response['attributeScores']['PROFANITY']['summaryScore']['value']
    threat = response['attributeScores']['THREAT']['summaryScore']['value']

    print("toxicity score of output: ", toxicity)
    print("severe toxicity score of output: ", severe_toxicity)
    print("identity attack score of output: ", identity_attack)
    print("insult score of output: ", insult)
    print("profanity score of output: ", profanity)
    
    dict = {
        "toxicity": toxicity,
        "severe_toxicity": severe_toxicity,
        "identity_attack": identity_attack,
        "insult": insult,
        "profanity": profanity,
        "threat": threat,
    }

    return dict

def perspectiveapi_3(text):
    # measure toxicity socre using perspectiveAPI
    client = discovery.build(
    'commentanalyzer',
    'v1alpha1',
    discoveryServiceUrl = DISCOVERY_URL,
    developerKey = 'AIzaSyCOrzGTUUTdwEtVSPiepSb7wt4Rooy5BfM',
    static_discovery = False,
    )

    analyze_request = {
        'comment': { 'text': text },
        'requestedAttributes': {
            'TOXICITY': {},
            'SEVERE_TOXICITY': {},
            'IDENTITY_ATTACK': {},
            'INSULT': {},
            'PROFANITY': {},
            'THREAT': {},
        }
    }

    response = client.comments().analyze(body=analyze_request).execute()
    toxicity = response['attributeScores']['TOXICITY']['summaryScore']['value']
    severe_toxicity = response['attributeScores']['SEVERE_TOXICITY']['summaryScore']['value']
    identity_attack = response['attributeScores']['IDENTITY_ATTACK']['summaryScore']['value']
    insult = response['attributeScores']['INSULT']['summaryScore']['value']
    profanity = response['attributeScores']['PROFANITY']['summaryScore']['value']
    threat = response['attributeScores']['THREAT']['summaryScore']['value']

    print("toxicity score of output: ", toxicity)
    print("severe toxicity score of output: ", severe_toxicity)
    print("identity attack score of output: ", identity_attack)
    print("insult score of output: ", insult)
    print("profanity score of output: ", profanity)
    
    dict = {
        "toxicity": toxicity,
        "severe_toxicity": severe_toxicity,
        "identity_attack": identity_attack,
        "insult": insult,
        "profanity": profanity,
        "threat": threat,
    }

    return dict