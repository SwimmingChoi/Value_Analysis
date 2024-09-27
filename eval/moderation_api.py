from openai import OpenAI

class ModerationAPI:

    def __init__(self, api_key, text):
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)
        self.text = text
        
    def flag(self):
            
        result = self.client.moderations.create(self.text)
        
        flagged = result.results[0].flagged
        
        return flagged
        
    def moderation(self):
        
        result = self.client.moderations.create(self.text)
        
        flagged = result.results[0].flagged
        
        sexual = result.results[0].categories.sexual
        hate = result.results[0].categories.hate
        harassment = result.results[0].categories.harassment
        self_harm = result.results[0].categories.self_harm
        harassment_threatening = result.results[0].categories.harassment_threatening
        hate_threatening = result.results[0].categories.hate_threatening
        self_harm_instructions = result.results[0].categories.self_harm_instructions
        self_harm_intent = result.results[0].categories.self_harm_intent
        sexual_minors = result.results[0].categories.sexual_minors
        violence = result.results[0].categories.violence
        violence_graphic = result.results[0].categories.violence_graphic
        
        categories = {
            "sexual": sexual,
            "sexual_minors": sexual_minors, 
            "hate": hate,
            "hate_threatening": hate_threatening,       
            "harassment": harassment,
            "harassment_threatening": harassment_threatening,
            "self_harm": self_harm,
            "self_harm_instructions": self_harm_instructions,
            "self_harm_intent": self_harm_intent,
            "violence": violence,
            "violence_graphic": violence_graphic
        }
        
        score_sexual = result.results[0].category_scores.sexual
        score_hate = result.results[0].category_scores.hate
        score_harassment = result.results[0].category_scores.harassment
        score_self_harm = result.results[0].category_scores.self_harm
        score_harassment_threatening = result.results[0].category_scores.harassment_threatening
        score_hate_threatening = result.results[0].category_scores.hate_threatening
        score_self_harm_instructions = result.results[0].category_scores.self_harm_instructions
        score_self_harm_intent = result.results[0].category_scores.self_harm_intent
        score_sexual_minors = result.results[0].category_scores.sexual_minors
        score_violence = result.results[0].category_scores.violence
        score_violence_graphic = result.results[0].category_scores.violence_graphic
        
        category_scores = {
            "sexual": score_sexual,
            "sexual_minors": score_sexual_minors, 
            "hate": score_hate,
            "hate_threatening": score_hate_threatening,       
            "harassment": score_harassment,
            "harassment_threatening": score_harassment_threatening,
            "self_harm": score_self_harm,
            "self_harm_instructions": score_self_harm_instructions,
            "self_harm_intent": score_self_harm_intent,
            "violence": score_violence,
            "violence_graphic": score_violence_graphic
        }
        
        dict = {
            "flagged": flagged,
            "categories": categories,
            "category_scores": category_scores,
        }
        
        return dict