"""
Job Agent - Job Posting Analysis and Company Research

This agent is responsible for extracting and analyzing job-related information:
1. Extracts structured job requirements from URLs (web scraping)
2. Extracts structured job requirements from pasted text
3. Researches company information using web search
4. Returns structured data for CV tailoring and cover letter writing

Architecture:
- Uses tool calling pattern (can be invoked by supervisor agent)
- Structured output with Pydantic models for data consistency
- Real web search integration (Tavily) for company research
- Interactive mode for standalone testing
- Returns JSON strings for easy integration

Tools:
- retrieve_web: Extract job info from URLs (SSRF-safe fetch + LLM extraction)
- retrieve_text: Extract job info from pasted text (LLM extraction)
- collect_company_info: Research company using Tavily search (web search + LLM extraction)

Data Models:
- JobRequirements: Structured job posting data (title, skills, responsibilities, qualifications, etc.)
- CompanyInfo: Structured company data (values, culture, mission, industry, etc.)
"""

from langchain.agents import create_agent
from langchain_core.tools import tool
from app.openai_llm import chat_openai_mini
from langchain_core.prompts import ChatPromptTemplate
from langchain_tavily import TavilySearch

from app.config import get_settings
from app.services.url_safety import fetch_public_page

# Import models from centralized schemas
from app.models.schemas import JobRequirements, CompanyInfo

import dotenv

dotenv.load_dotenv()

# --------------------------------------------------------------------------- #
# Shared extraction prompt
#
# Postings routinely phrase requirements as OR-alternatives ("Java and
# WildFly/Quarkus/Spring Boot") or explicitly say some skills are trainable
# on the job ("was Du noch nicht kennst, lernst Du bei uns"). A flat
# required_skills list that ignores both reads those postings far too
# strictly: a candidate with Java + Spring Boot only, correctly missing
# WildFly and Quarkus, would otherwise show 2 of 4 "genuine gaps" for a
# requirement they in fact fully satisfy. The extraction step is the only
# place this nuance can be captured - by the time it's a flat skill list,
# the "or" and "we'll train you" framing is already lost.
# --------------------------------------------------------------------------- #

_JOB_EXTRACTION_SYSTEM_PROMPT = """Extract structured job information from the provided job posting. \
Focus on identifying all required skills, responsibilities, qualifications, and key requirements.

Skill requirements need careful reading, not a flat keyword dump:

1. OR-alternatives: when a posting lists a group of interchangeable options \
(separated by "/", "or", "oder", or phrasing like "one of the following"), \
that is ONE requirement satisfied by ANY ONE of them - not every item in the \
list. Example: "Java and WildFly/Quarkus/Spring Boot" is two requirements: \
Java (mandatory), and a JVM framework (any one of WildFly, Quarkus, Spring \
Boot). Record this as a group in `required_skill_groups`: each inner list is \
one requirement, its items are the interchangeable alternatives. A required \
skill with no stated alternative is still its own single-item group (e.g. \
["Java"]). Put the union of every group's members into `required_skills` as \
before, for display - `required_skill_groups` is what makes the OR meaning \
explicit for scoring.
2. Classify by INTENT, not by keyword-spotting a skill name. For every skill \
mentioned, read the actual sentence around it and ask: does this sentence \
assert the candidate already POSSESSES the skill today, or does it merely \
express willingness, aspiration, curiosity, or a future learning opportunity \
around it? Only the first is a hard requirement. Phrasing patterns that mean \
"optional, not required" even when the skill name itself sounds important: \
   a. Employer will teach it: "was Du noch nicht kennst, lernst Du bei uns", \
      "training provided", "you'll learn on the job", "no prior experience \
      needed with X", "we'll get you up to speed on X".\
   b. Candidate's willingness/enthusiasm is what's asked for, not possession: \
      "eager to learn Java", "excited to work with X", "willingness to learn \
      X", "curiosity for X", "you enjoy picking up new tools like X", \
      "interest in X is a plus". These are asking for an attitude, not \
      proof of the skill - the skill named is a bonus if already held, never \
      a day-one must-have.\
   c. Explicit softeners anywhere near a skill: "nice to have", "a plus", \
      "bonus", "ideally", "would be great if", "no experience necessary".\
   Any skill whose sentence matches (a), (b), or (c) belongs in \
   `preferred_skills`, not `required_skills` / `required_skill_groups` - even \
   when it's named in the same sentence, or the same skill/framework list, as \
   a genuinely required skill. Conversely, do not soften a skill that IS \
   stated as a firm possession requirement ("must have X", "several years of \
   X", "solid experience with X", "proficiency in X") just because a nearby, \
   different skill happens to be phrased optionally.
3. Only skills the sentence asserts the candidate must already possess on \
day one belong in `required_skills` / `required_skill_groups`. Everything \
else - nice-to-haves, bonus skills, trainable skills, "eager to learn" \
skills - belongs in `preferred_skills`. When genuinely unsure after reading \
the sentence's intent, prefer `preferred_skills` - overstating a requirement \
does more damage to a fair compatibility score than understating one.
4. Job-position framing: read the exact title and how the role is described \
(Werkstudent / working student, intern, junior, mid, senior, lead). The same \
technology list is not the same bar on a student posting as on a Senior \
Engineer posting. Do not inflate a junior/student role's listed stack into \
hard day-one must-haves; do not drop a senior role's stated must-haves \
because those tools also appear on junior postings.
5. Languages and education are eligibility requirements, not skills. Put \
spoken-language requirements (with the stated level, e.g. "German (C1)", \
"English (fluent)", "Deutsch verhandlungssicher") into `required_languages` \
only when the posting requires them; a language that is merely "a plus" is \
not required and goes nowhere. Put degree, field-of-study, and enrollment \
requirements ("enrolled student in Computer Science or a related field", \
"Bachelor's in Engineering", "eingeschriebene/r Student/in der Informatik") \
into `education_requirements`. Never list a language, a degree, or "student" \
in `required_skills`, `preferred_skills`, or `required_skill_groups`.

If the posting names a contact person, hiring manager, or recruiter to \
address applications to, capture their name in recipient_name; otherwise \
leave recipient_name empty."""


@tool
def retrieve_web(web_urls: list[str]) -> str:
    """
    Takes one or more URLs and extracts all job related information from them.
    Can accept a single URL as a list with one element, or multiple URLs.
    
    Args:
        web_urls: List of job posting URLs to extract information from
        
    Returns:
        JSON string with structured job requirements information
    """
    # Fetched through the SSRF-safe fetcher: only public hosts, redirects
    # re-checked hop by hop, time and size capped (WebBaseLoader fetched
    # whatever URL it was given, internal addresses included).
    user_agent = get_settings().user_agent
    text_content = "\n\n".join(fetch_public_page(url, user_agent) for url in web_urls)
    llm = chat_openai_mini(temperature=None)
    prompt = ChatPromptTemplate.from_messages([
       ("system", _JOB_EXTRACTION_SYSTEM_PROMPT),
       ("user", "Here are the pages necessary for application {pages}")
    ])
    structured_llm = llm.with_structured_output(JobRequirements)
    chain = prompt | structured_llm
    result = chain.invoke({"pages": text_content})
    return result.model_dump_json(indent=2)

@tool
def retrieve_text(info: str) -> str:
    """
    Takes the raw text input and extract all job related information.
    
    Args:
        info: Raw text of job posting or job description
        
    Returns:
        JSON string with structured job requirements information
    """
    llm = chat_openai_mini(temperature=None)
    prompt = ChatPromptTemplate.from_messages([
       ("system", _JOB_EXTRACTION_SYSTEM_PROMPT),
       ("user", "Here is the text necessary for application {text}")
    ])
    structured_llm = llm.with_structured_output(JobRequirements)
    chain = prompt | structured_llm
    result = chain.invoke({"text": info})
    return result.model_dump_json(indent=2)

@tool
def collect_company_info(company_name: str) -> str:
    """
    Executes an online search to collect basic information of the company, e.g. company's history, goals...
    
    Args:
        company_name: Name of the company to research
        
    Returns:
        JSON string with structured company information
    """
    # Init search tool
    search = TavilySearch(max_results=5)
    # Perform search with query
    query = f"What are {company_name}'s company values, culture, and mission statement?"
    # search_results returns a list of dicts with "title", "url", "content" keys
    search_results = search.invoke(query)
    # Format results for LLM
    formatted_results = "\n\n---\n\n".join([
        f"Title: {result.get('title', 'N/A')}\nURL: {result.get('url', 'N/A')}\nContent: {result.get('content', 'N/A')}"
        for result in search_results
    ])

    llm = chat_openai_mini(temperature=None)
    structured_llm = llm.with_structured_output(CompanyInfo)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract structured company information from the search results. Focus on identifying company values, culture, mission statement, industry, company size, products/services, and recent news that would be relevant for someone applying to this company."),
        ("user", "Company name: {company_name}\n\nSearch results:\n{search_results}")
    ])

    chain = prompt | structured_llm
    result = chain.invoke({"company_name": company_name, "search_results": formatted_results})
    return result.model_dump_json(indent=2)

llm = chat_openai_mini(temperature=None)
system_prompt = """You are a job analysis assistant. Your role is to help users gather comprehensive information about job postings and companies.

Your available tools:
1. retrieve_web - Extract job information from URLs
2. retrieve_text - Extract job information from pasted text
3. collect_company_info - Research company information online

Workflow:
1. When given a job posting URL, use retrieve_web to extract job requirements
2. When given job description text, use retrieve_text to extract job requirements
3. When asked about a company, use collect_company_info to research company details
4. Present extracted information clearly and ask if the user needs additional research

Be thorough and ensure all relevant information is captured for job applications."""

# Create Job Agent
agent = create_agent(
    model = llm,
    tools = [
        retrieve_web,
        retrieve_text,
        collect_company_info
        ],
    system_prompt = system_prompt
)


def run_interactive_job_analysis(initial_input: str = None):
    """
    Run job analysis with interactive user conversation via terminal.
    
    The agent can:
    1. Extract job requirements from URLs
    2. Extract job requirements from pasted text
    3. Research company information
    4. Answer questions about the job or company
    """
    print("=" * 70)
    print("JOB ANALYSIS AGENT - Interactive Mode")
    print("=" * 70)
    print("\nI can help you analyze job postings and research companies.")
    print("\nExamples:")
    print("  - 'Analyze this job URL: https://...'")
    print("  - 'Research Google as a company'")
    print("  - Paste job description text directly")
    print("\nType 'quit' to exit at any time.\n")
    
    # Initialize conversation
    if initial_input:
        messages = [{"role": "user", "content": initial_input}]
        print(f"Starting with: {initial_input}\n")
    else:
        user_input = input("You: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            return
        messages = [{"role": "user", "content": user_input}]
    
    # Conversation loop
    max_turns = 15
    turn = 0
    
    while turn < max_turns:
        turn += 1
        print(f"\n{'='*70}")
        print(f"Turn {turn}")
        print(f"{'='*70}")
        
        # Agent's turn
        print("\n🤖 Agent is processing...\n")
        try:
            result = agent.invoke({"messages": messages})
            agent_response = result["messages"][-1].content
            
            # Add agent's response to history
            messages.append({"role": "assistant", "content": agent_response})
            
            # Display agent's response
            print(f"Agent:\n{agent_response}\n")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again with different input.\n")
            break
        
        # User's turn
        print(f"{'-'*70}")
        user_input = input("You: ").strip()
        
        # Allow user to quit
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        # Add user's response to history
        messages.append({"role": "user", "content": user_input})
    
    if turn >= max_turns:
        print(f"\n⚠️  Reached maximum turns ({max_turns}). Ending conversation.")
    
    return {
        "messages": messages,
        "turns": turn
    }


# CLI for testing
if __name__ == "__main__":
    import sys
    
    print("\n" + "="*70)
    print("JOB AGENT - Interactive Terminal Mode")
    print("="*70 + "\n")
    
    # Get initial input from command line or prompt user
    if len(sys.argv) > 1:
        initial_input = " ".join(sys.argv[1:])
    else:
        initial_input = None
    
    # Run interactive analysis
    try:
        run_interactive_job_analysis(initial_input)
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Exiting.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()