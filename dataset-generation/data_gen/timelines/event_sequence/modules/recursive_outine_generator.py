from typing import List, Dict

from data_gen.llm.critiques.base_critique import BaseCritique
from data_gen.llm.critiques.output_format_critique import OutputFormatCritique
from data_gen.llm.modules.parsable_base_module import ParsableBaseModule
from data_gen.llm.prompting.modules.nested_parsable_output_prompt import BasicNestedXMLParser
from data_gen.llm.prompting.parsable_prompt import ParsablePrompt
from data_gen.llm.verifier.named_unified_output_verifier import NamedUnifiedOutputVerifier
from data_gen.llm.wrapper.base_llm_wrapper import BaseLLMWrapper
from data_gen.timelines.event_sequence.modules.fictive_entities.entity_critiques.crazy_topic_critique import \
    CrazyTopicCritique

EXPECTED_OUTPUT_FORMAT = """
The output format is incorrect. Please output the results in the following format:
<results>
<date>Month Day, Year</date>
<outline>
<storyitem>First story item</storyitem>
<storyitem>Second story item</storyitem>
...
<storyitem>Last story item</storyitem>
</outline>
<causal>
<edge>
<cause_text>[Paste the exact cause sentence here]</cause_text>
<effect_text>[Paste the exact effect sentence here]</effect_text>
<relation>causes|enables|prevents</relation>
<explanation>...</explanation>
</edge>
...
</causal>
</results>
""".strip()


class OutlineGenerationModule(ParsableBaseModule):
    """
    This module generates the fictional outline.
    """

    def _create_critiques(self) -> List[BaseCritique]:
        return [CrazyTopicCritique('story_item', 'list')]

    def _create_formatting_critique(self, parsers: List[BasicNestedXMLParser]) -> BaseCritique:
        return OutputFormatCritique('format-seed-outline', parsers, EXPECTED_OUTPUT_FORMAT)

    def __init__(self, llm: BaseLLMWrapper, name: str, instruction_name: str, num_story_items: int):
        super().__init__(
            llm,
            name,
            instruction_name,
            get_instructions(instruction_name),
            max_critiques=5
        )
        self.num_story_items: int = num_story_items

    def _preprocess_values(self, values) -> Dict:
        values['num_storyitems'] = self.num_story_items
        values['history_xml'] = '\n'.join(values['histories'])
        return values

    def _get_verifiers(self) -> List[NamedUnifiedOutputVerifier]:
        return []

    def _get_parsers(self) -> List[BasicNestedXMLParser]:
        return [
            BasicNestedXMLParser('story_item', './/storyitem', is_object=False, result_node='results', remove_node='scratchpad'),
            BasicNestedXMLParser('date', 'date', is_object=False, to_single=True, result_node='results', remove_node='scratchpad'),
            BasicNestedXMLParser('causal_edges', './/causal/edge', is_object=True, result_node='results', remove_node='scratchpad')
        ]

    def get_file_name(self, prompt: ParsablePrompt, values: Dict):
        summary = values['EVENT_SUMMARY_FOR_NAME'].lower().replace(' ', '-')
        node_idx = values['CREATED_AT']
        return f'N{node_idx:02d}-{self.name}-{summary}_{self.instruction_name}.json'


def get_instructions(version: str) -> str:
    
    if version == 'v5':
        out: str = INSTRUCTIONS_V5
    else:
        raise ValueError(version)
    return out.strip()

INSTRUCTIONS_V5 = """
You are an AI assistant tasked with generating an outline for a fictional event. Your goal is to create a realistic, entirely fictional event that does not overlap with real-world named entities or known fictional named entities. Follow these instructions carefully:

First, review the list of already known fictional named entities of this fictional world:

<known_entities>
<LOCATIONS>
{{LOCATIONS_XML}}
</LOCATIONS>

<PERSONS>
{{PERSONS_XML}}
</PERSONS>

<ORGANIZATIONS>
{{ORGANIZATIONS_XML}}
</ORGANIZATIONS>

<PRODUCTS>
{{PRODUCTS_XML}}
</PRODUCTS>

<ARTS>
{{ARTS_XML}}
</ARTS>

<EVENTS>
{{EVENTS_XML}}
</EVENTS>

<BUILDINGS>
{{BUILDINGS_XML}}
</BUILDINGS>

<MISCELLANEOUS>
{{MISCELLANEOUSS_XML}}
</MISCELLANEOUS>
</known_entities>
Next, review the outline of previous events that have occurred in this fictional world:

<history>
{{HISTORY_XML}}
</history>

Now, consider the following information about the new event you need to generate as a continuation of the past events:

Date: {{PROVIDED_DATE}}

Event Summary: {{EVENT_SUMMARY}}

Genre: {{GENRE}}

Follow these guidelines to generate the event outline:

1. Create an entirely fictional event based on the given genre, event summary, and history of previous events. The event must be realistic but must not reference any existing real-world or known fictional named entities.
2. Invent new named entities as needed, ensuring they don't exist in the real world or in existing works of fiction. When creating names, use unique combinations unlikely to match real named entities.
3. Construct the outline using short, concise, factual, and objective statements. Each statement must discuss only one fact or sub-event, structured sequentially in a logical temporal order when applicable.
4. Ensure all statements form a coherent outline.
5. Output each statement within a <storyitem> tag.
6. Generate exactly {{NUM_STORYITEMS}} distinct story items.
7. Ensure logical progression, with each statement following chronologically when applicable. Include a mix of main events, reactions, consequences, and contextual information.
8. Make storyitems as atomic as possible, communicating only a single piece of relevant information per item. Do not merge multiple pieces of information into one storyitem.
9. Ensure the story sounds realistic without explicitly stating it's fictional.
10. Maintain consistency with the provided <history> that discusses past events fictional events. The outline must logically follow chronological events described in the history.
11. Incorporate some or all of the provided named entities in your outline. Ensure that any mention of these entities is consistent with the information you have about the named entity. You may introduce additional fictional entities as needed, but they must not conflict with the existing ones.
12. When referencing any named entities from the provided inputs, maintain consistency in their descriptions and roles within the story.
13. If no date is provided: Generate a complete date for the event, including the year. The date should be formatted as "year-month-day" (e.g., "2024-12-03" or "2025-06-13"). This date should be consistent with the timeline established in the <history>.
14. If a date is provided: Use the provided date.
15. The outline can include quotes from the named entities where applicable.
16. Do not repeat the information from the previous events from the <history>.
17. Refer to all named entities (the new named entities and the known named entities) by their full "name" property. DO NOT refer to the named entities using the ID.
18. Make sure that you refer to all named entities within each storyitem per full name at least once. DO NOT use pronouns to refer to a named entity from the previous story item.
19. Think about the content that is appropriate for the event summary given the genre, provided history: Think about which dimensions align with all of those, and sound like a realistic event.
20. Generate explicit causal relations in a separate <causal> section using the exact sentences:
    - For each edge, set <cause_text> to the exact cause sentence (as it appears in <storyitem>) and <effect_text> to the exact effect sentence.
    - Only add an edge when there is a clear, specific causal or enabling relation; do not add edges merely because of temporal order.
    - Keep relations concise; use simple labels like causes|enables|prevents and, optionally, an <explanation>.
    - Limit the number of edges to a small, coherent set (e.g., 1–4) that best captures the causal backbone.
    - Every cause_text must refer to a sentence that appears earlier in the outline than the effect_text; no self-links; no backward-in-time links.

Your output should be formatted as follows:
<scratchpad>[Your thoughts go here]</scratchpad>
<results>
<date>year-month-day</date>
<outline>
<storyitem>[Sentence 1]</storyitem>
<storyitem>[Sentence 2]</storyitem>
...
</outline>
<causal>
<edge>
<cause_text>[Paste the exact cause sentence here]</cause_text>
<effect_text>[Paste the exact effect sentence here]</effect_text>
<relation>causes|enables|prevents</relation>
<explanation>...</explanation>
</edge>
...
</causal>
</results>

IMPORTANT:
- The event must be entirely realistic, even though it is fictional. Do not include any science fiction or fantasy elements. The story should read like a plausible current event.
- Do not use any galactic events. The fictional world should be similar to our world but not about galaxies or outer space.
- Each story item must only discuss one fact or subevent. Ensure that each story item is specific, concise, and focused on a single piece of information.
- Begin your response with <results> and end it with </results>. Do not include any text outside of these tags.
- Do not exaggerate the outline. Avoid using words like "groundbreaking", "worldwide", "global". Keep the outline and the scope and influence of the event realistic.
- Do not create outlines with global or national impact unless the genre specifically requires it. Instead, focus on smaller or local developments.
- Do not focus on technological discoveries or topics like AI tools, virtual reality, augmented reality, 3D-modelling, quantum computing, etc. You may include such topics only if they are HIGHLY relevant to the genre {{GENRE}} AND the provided history of events.
- Focus on realistic, meaningful outlines with specific details and events that align with typical, realistic scenarios of the genre {{GENRE}}.
- Encode causal relations only via the <causal> section using <edge> elements with <cause_text> and <effect_text> that refer to the exact sentences in <outline>.
- Ensure that no cause–effect link goes against chronological order: every cause_text must refer to a story item that happens earlier in the outline than the story item whose effect_text you are specifying.

Remember:
The outline should center on a fictional but realistic event, keeping its scale aligned with the event summary and provided <history> without exaggerating its impact. Rather than overstating the event's significance, the outline should stay within the scope appropriate to the genre, provided history, and provided summary. When in doubt, focus on detailed, localized developments instead of amplifying global effects.

Ensure that the outline is coherent, follows a logical sequence, and offers a unique perspective on the given event while maintaining consistency with the provided background information.
"""