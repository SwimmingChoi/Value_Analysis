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