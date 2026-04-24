import os
import re
import json
import requests
import argparse
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich import print as rprint

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
console = Console()

SYSTEM_PROMPT = """You are a research agent. Your job is to research a topic thoroughly and write a detailed report.

You have access to two tools:
- search(query): searches the web and returns results
- fetch(url): fetches the content of a webpage

To use a tool, respond ONLY with JSON in this exact format:
{"tool": "search", "input": "your search query"}
{"tool": "fetch", "input": "https://example.com"}

When you have gathered enough information, respond with:
{"tool": "done", "input": ""}

STRICT RULES:
1. You MUST search at least 2 times before calling done
2. You MUST fetch at least 3 URLs before calling done
3. After every search, pick the 2 most relevant URLs and fetch them
4. Never write the report from snippets alone — always fetch the full pages first
5. Do NOT mix text and JSON. Either call a tool (JSON only) or write the final report (text only)."""
# ── TOOLS ──────────────────────────────────────────────

def search(query):
    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": os.getenv("SERPER_API_KEY"),
        "Content-Type": "application/json"
    }
    body = {"q": query, "num": 5}
    response = requests.post(url, headers=headers, json=body)
    results = response.json().get("organic", [])
    formatted = []
    for r in results:
        formatted.append(f"Title: {r['title']}\nURL: {r['link']}\nSnippet: {r['snippet']}")
    return "\n\n".join(formatted)


def fetch(url):
    try:
        response = requests.get(url, timeout=10)
        text = response.text
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:3000]
    except Exception as e:
        return f"Failed to fetch: {str(e)}"


# ── AGENT LOOP ─────────────────────────────────────────

def ask(messages):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=4096,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages
    )
    return response.choices[0].message.content


def run_agent(topic, depth=8):
    console.print(Panel(f"[bold cyan]🔍 Researching: {topic}[/bold cyan]", border_style="cyan"))

    messages = [{"role": "user", "content": f"Research this topic: {topic}"}]
    max_iterations = depth * 3
    step = 1
    collected_content = []  # we collect everything ourselves

    for i in range(max_iterations):
        with console.status(f"[yellow]Thinking...[/yellow]", spinner="dots"):
            reply = ask(messages)

        try:
            json_match = re.search(r'\{.*\}', reply, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found")

            tool_call = json.loads(json_match.group())
            tool = tool_call["tool"]
            inp  = tool_call["input"]

            if tool == "done":
                console.print(f"\n[bold green]✅ Research complete. Writing report...[/bold green]\n")
                break

            elif tool == "search":
                console.print(f"[bold blue]  [{step}] 🔎 Searching:[/bold blue] {inp}")
                result = search(inp)
                collected_content.append(f"### Search: {inp}\n{result}")
                messages.append({"role": "assistant", "content": reply})
                messages.append({"role": "user", "content": f"Search results:\n{result}"})

            elif tool == "fetch":
                console.print(f"[bold magenta]  [{step}] 🌐 Fetching:[/bold magenta] {inp}")
                result = fetch(inp)
                collected_content.append(f"### Page: {inp}\n{result}")
                messages.append({"role": "assistant", "content": reply})
                messages.append({"role": "user", "content": f"Page content:\n{result}"})

            step += 1

        except (json.JSONDecodeError, ValueError):
            break

    # fresh clean API call just for writing — no messy tool history
    all_content = "\n\n---\n\n".join(collected_content)
    with console.status("[yellow]Writing report...[/yellow]", spinner="dots"):
        report = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=4096,
            messages=[
                {"role": "system", "content": "You are a professional research writer. Write clear, detailed, well-structured reports in markdown."},
                {"role": "user", "content": f"""Based on the following research collected on the topic '{topic}', write a detailed markdown report.

RULES:
- Write in proper prose paragraphs
- Use markdown headings to organize sections
- Synthesize the information in your own words
- Only put URLs in a 'Sources' section at the very end
- Do NOT list raw links in the body

RESEARCH COLLECTED:
{all_content}"""}
            ]
        ).choices[0].message.content

    return report


# ── RUN ────────────────────────────────────────────────

# ── CLI ARGS ───────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("topic", nargs="?", default=None, help="Research topic")
parser.add_argument("--d", type=int, default=8, help="Research depth (default: 8)")
args = parser.parse_args()

# get topic from CLI or prompt
topic = args.topic or input("Enter research topic: ")
depth = args.d

console.print(f"[dim]Depth set to {depth} iterations[/dim]")

report = run_agent(topic, depth)

# pretty print the report
console.print("\n")
console.print(Panel("[bold white]FINAL REPORT[/bold white]", border_style="green"))
console.print(Markdown(report))

# timestamped filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
safe_topic = topic.replace(" ", "_")[:30]  # trim long topics
filename = f"report_{safe_topic}_{timestamp}.md"

with open(filename, "w") as f:
    f.write(f"# {topic}\n*Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}*\n\n")
    f.write(report)

console.print(f"\n[bold green]📄 Saved to:[/bold green] [cyan]{filename}[/cyan]")