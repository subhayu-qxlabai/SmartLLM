from pathlib import Path
from typing import Callable

from models.outputs import ExtractEntryWithFunction
from models.extractor import ExtractorDatasetRowOutput, ExtractSchemaEntry, ExtractorIO
from helpers.utils import try_json_dumps
from helpers.call_openai import call_openai_api
from dataset_gen.base.json_generator import JSONGenerator

DEFAULT_DUMP_DIR = Path("generated/step_output")


class ExtractGenerator(JSONGenerator):
    def __init__(
        self,
        dump_dir: str | Path = DEFAULT_DUMP_DIR,
        file_prefix: str = "x",
        system_prompt: str = None,
        example_messages: list[dict[str, str]] = [],
        openai_func: Callable = call_openai_api,
        max_tokens: int = 4000,
        verbose: bool = True,
    ):
        system_prompt = system_prompt or (
            "When generating function outputs, make sure that the values specified in the schema are present in the output data."
            "In the input, the `output_language` key denotes in which language the function_output should be"
            "Each function should return a JSON response with the following structure: \n"
            '{\n    "function_output": {\n      "FUNCTION_NAME": {\n        // Output data for the function in the specified language\n      },\n      // Additional function outputs if applicable\n    },\n    "extracted_schema": {\n      // Extracted values from the function output corresponding to the schema and in the specified language\n    }\n  }\n\n'
            "Please ensure that the 'function_output' key contains the outputs of each function, and the 'extracted_schema' key includes the values extracted from the function outputs that match the schema."
        )
        example_messages = example_messages or [
            {
                "role": "user",
                "content": '{"functions": [{"id": "function_1", "name": "MEDICAL_RESEARCH_QUERY", "parameters": [{"name": "query", "value": "most prevalent bacteria in failed root canal treatments"}, {"name": "databases", "value": ["PubMed", "Cochrane Library", "MEDLINE"]}, {"name": "include_abstracts", "value": 1}]}], "schema": [{"name": "bacteria_name", "type": "string", "description": "The name of the most prevalent bacteria in failed root canal treatments."}], "output_language": "hindi"}',
            },
            {
                "role": "assistant",
                "content": '{"function_output": {"MEDICAL_RESEARCH_QUERY": {"research_summary": "रूट कैनाल उपचारों के असफलता के साहित्य की व्यापक समीक्षा में एंटरोकोकस फेकैलिस को इन मामलों में पाए जाने वाले सबसे प्रचलित बैक्टीरिया के रूप में पहचाना गया है। यह निष्कर्ष पबमेड, कोक्रेन लाइब्रेरी और मेडलाइन डेटाबेसों में फैले हुए कई शोध पत्रों और समीक्षाओं के आँकड़ों से निकाला गया था। हमारे खोज मानदंडों में सारांशों का समावेश ने हाल के और पुराने अध्ययनों दोनों का पूरी तरह से विश्लेषण करने की अनुमति दी, जिससे असफल रूट कैनाल उपचारों में बैक्टीरियल प्रवलता की व्यापक समझ सुनिश्चित हुई।"}}, "extracted_schema": {"bacteria_name": "एंटरोकोकस फेकैलिस"}}',
            },
            {
                "role": "user",
                "content": '{"functions": [{"id": "function_1", "name": "FASHION_SEARCH", "parameters": [{"name": "keywords", "value": "Hilary Duff shopping clothes accessories"}]}, {"id": "function_2", "name": "FASHION_SEARCH", "parameters": [{"name": "keywords", "value": "Hilary Duff social media shopping"}]}], "schema": [{"name": "shopping_locations", "type": "array", "description": "List of places where Hilary Duff shops for clothes or accessories."}], "output_language": "english"}',
            },
            {
                "role": "assistant",
                "content": '{"function_output":{"FASHION_SEARCH":[{"keywords":"Hilary Duff shopping clothes accessories","search_results":"Hilary Duff has been recently spotted shopping at Zara and H&M for casual wear. For accessories, she favors boutique stores like The Little Market."},{"keywords":"Hilary Duff social media shopping","search_results":"On social media, Hilary Duff has shared multiple purchases from online platforms such as Etsy for unique items and ASOS for trendy clothes."}]},"extracted_schema":{"shopping_locations":["Zara", "H&M", "The Little Market", "Etsy", "ASOS"]}}',
            },
            {
                "role": "user",
                "content": '{"functions": [{"id": "function_1", "name": "ACADEMIC_RESEARCH", "parameters": [{"name": "query", "value": "gay Republicans reconciliation sexual orientation political beliefs"}, {"name": "fields_of_study", "value": ["political science", "sociology", "psychology"]}, {"name": "number_of_results", "value": 10}, {"name": "sort_by", "value": "relevance"}]}, {"id": "function_2", "name": "NEWS_API", "parameters": [{"name": "keywords", "value": ["gay Republicans", "sexual orientation", "political beliefs"]}, {"name": "date_range", "value": {"start_date": "2022-01-01", "end_date": "2023-01-01"}}, {"name": "language", "value": "french"}]}, {"id": "function_3", "name": "INTERVIEW_DATABASE", "parameters": [{"name": "keywords", "value": ["gay Republicans", "interviews"]}, {"name": "date_range", "value": {"start_date": "2022-01-01", "end_date": "2023-01-01"}}]}], "schema": [{"name": "articles", "type": "array", "description": "Collection of academic and news articles on gay Republicans."}, {"name": "interviews", "type": "array", "description": "Collection of interviews with gay Republicans."}], "output_language": "french"}',
            },
            {
                "role": "assistant",
                "content": '{"function_output": {"ACADEMIC_RESEARCH": {"research_articles": [{"title": "Concilier l\'orientation sexuelle et les convictions politiques chez les républicains homosexuels", "authors": ["John Doe", ""], "journal": "Revue de Psychologie Politique", "year": 2022, "abstract": "Cette étude explore comment les républicains homosexuels concilient leur orientation sexuelle avec leurs convictions politiques. En utilisant des analyses qualitatives, nous révélons les complexités de la réconciliation identitaire au sein de cette communauté."}, {"title": "L\'intersection de l\'idéologie politique et de l\'identité sexuelle", "authors": ["Maxwell Irving"], "journal": "Sociologie de l\'Identité", "year": 2022, "abstract": "Un examen des défis personnels et sociétaux auxquels sont confrontés les membres homosexuels des partis conservateurs, en soulignant le rôle des systèmes de soutien social et des mécanismes de coping interne."}]}, "NEWS_API": {"news_articles": [{"title": "La lutte silencieuse : des Républicains homosexuels s\'expriment", "date": "20 mai 2022", "source": "Le Politico", "url": "https://thepolitico.example.com/gay-republicains-lutte"}, {"title": "Identité et Appartenance : L\'Expérience du Républicain Gay", "date": "15 septembre 2022", "source": "Actualités nationales", "url": "https://nationalnews.example.com/gay-republican-experience"}]}, "INTERVIEW_DATABASE": {"interviews": [{"title": "Trouver ma voix : le parcours d\'un républicain homosexuel", "date": "4 juillet 2022", "platform": "InterviewsEnLigne", "url": "https://interviewsonline.example.com/gay-republican-journey"}, {"title": "Entre deux mondes : naviguer entre la politique et l\'identité", "date": "22 novembre 2022", "platform": "Entretiens en vedette", "url": "https://spotlightinterviews.example.com/politics-identity-gay-republican"}]}}, "extracted_schema": {"articles": [{"title": "Réconcilier l\'orientation sexuelle et les croyances politiques parmi les Républicains homosexuels", "authors": ["John Doe", ""], "journal": "Revue de Psychologie Politique", "year": 2022, "abstract": "Cette étude explore comment les républicains homosexuels concilient leur orientation sexuelle avec leurs croyances politiques. En utilisant des analyses qualitatives, nous révélons les complexités de la réconciliation d\'identité au sein de cette communauté."}, {"title": "L\'intersection de l\'idéologie politique et de l\'identité sexuelle", "authors": ["Maxwell Irving"], "journal": "Sociologie de l\'identité", "year": 2022, "abstract": "Un examen des défis personnels et sociétaux auxquels sont confrontés les membres homosexuels des partis conservateurs, en mettant l\'accent sur le rôle des systèmes de soutien social et des mécanismes d\'adaptation internes."}, {"title": "La lutte silencieuse : Les Républicains homosexuels s\'expriment", "date": "20 mai 2022", "source": "Le Politico", "url": "https://thepolitico.example.com/gay-republicains-lutte"}, {"title": "Identité et appartenance : L\'expérience des républicains homosexuels", "date": "15 septembre 2022", "source": "Actualités nationales", "url": "https://nationalnews.example.com/gay-republican-experience"}], "interviews": [{"title": "Trouver ma voix : Le parcours d\'un républicain homosexuel", "date": "4 juillet 2022", "platform": "InterviewsEnLigne", "url": "https://interviewsonline.example.com/gay-republican-journey"}, {"title": "Entre deux mondes : naviguer entre la politique et l\'identité", "date": "22 novembre 2022", "platform": "", "url": "https://spotlightinterviews.example.com/politics-identity-gay-republican"}]}}',
            },
            {
                "role": "user",
                "content": '{"functions": [{"id": "function_1", "name": "PEACE_TREATY_ANALYSIS", "parameters": [{"name": "treaty_name", "value": "2023 peace initiatives"}, {"name": "context", "value": "Identify and analyze the latest peace treaties and agreements signed or proposed in the year 2023."}]}, {"id": "function_2", "name": "PROMOTE_PEACE_INITIATIVES", "parameters": [{"name": "region", "value": "global"}]}], "schema": [{"name": "summary_of_initiatives", "type": "string", "description": "A combined description of the latest peace initiatives and agreements."}], "output_language": "turkish"}',
            },
            {
                "role": "assistant",
                "content": '{"function_output": {"PEACE_TREATY_ANALYSIS": {"analysis": "2023\'te, birkaç önemli barış antlaşması ve girişim, küresel huzur ve anlayışa doğru önemli bir adım olarak kayda geçmiştir. Özellikle, \'Küresel Uyum Mutabakatı\' uzun süredir devam eden toprak anlaşmazlıklarını diplomatik diyalog ve karşılıklı saygı yoluyla ele almayı amaçlıyor ve sürdürülebilir bir barışı teşvik etmeyi hedefliyor. Ayrıca, \'Birlik ve Uzlaşma Çerçevesi\' geçmiş çatışmaların yaralarını iyileştirmeye odaklanıyor, çatışan taraflar arasında diyalog ve anlayış için bir platform sağlıyor. Bu girişimler, dünyanın en kalıcı gerilimlerinden bazılarını çözme yolunda umut verici bir adım temsil ediyor."}, "PROMOTE_PEACE_INITIATIVES": {"promotion_activities": "Bu barış girişimlerini küresel olarak teşvik etmek için yapılan çabalar arasında \'Global Harmony Accord\' ve \'Birlik ve Uzlaşma Çerçevesi\'ne farkındalık ve destek artırmayı hedefleyen bir dizi sanal konferans bulunmaktadır. Ek olarak, sosyal medyayı kullanarak genç demografiyi etkilemeyi ve barış süreçlerine aktivizm ve destek teşvik etmeyi amaçlayan bir kampanya yürütülmektedir."}}, "extracted_schema": {"summary_of_initiatives": "2023 yılında, \'Küresel Uyum Antlaşması\' ve \'Birlik ve Uzlaşma Çerçevesi\', sırasıyla toprak anlaşmazlıklarını ele almayı ve geçmiş çatışmaların yaralarını sarmayı amaçlayan en dikkate değer barış girişimleri arasındaydı. Diplomatik diyalog, karşılıklı saygı ve anlayış yoluyla bu çabalar, küresel huzura doğru umut verici bir adım temsil etmektedir."}}',
            },
        ]
        super().__init__(
            dump_dir,
            file_prefix,
            system_prompt,
            example_messages,
            openai_func,
            max_tokens,
            verbose,
        )

    def generate(
        self,
        funcs_and_schemas: ExtractEntryWithFunction | str | dict,
        language: str = "english",
        dump: bool = False,
        validate: bool = True,
        strict: bool = False,
        *args,
        **kwargs,
    ):
        if isinstance(funcs_and_schemas, (str, dict)):
            funcs_and_schemas = self._to_model(
                funcs_and_schemas, ExtractEntryWithFunction
            )
        if not isinstance(funcs_and_schemas, ExtractEntryWithFunction):
            return None
        last_user_message = try_json_dumps(dict(sorted(
            (funcs_and_schemas.model_dump(mode="json")).items(),
            key=lambda x: x[0],
        ))|{"output_language": language})
        x_out = self.generate_response(last_user_message, dump)
        x_out_model: ExtractorDatasetRowOutput | None = self._to_model(
            x_out, ExtractorDatasetRowOutput
        )
        if not isinstance(x_out_model, ExtractorDatasetRowOutput):
            return None
        xio = x_out_model.to_extractor_io(
            [x.model_dump() for x in funcs_and_schemas.eschema]
        )
        return xio
