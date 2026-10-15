# AthenAI

AthenAI is an AI-powered study agent for students using Mediasite. It helps students study from their course video transcripts and from additional materials they choose to upload, such as notes, Word documents, PDFs, PowerPoint slides, and other class resources.

## What It Does

- Answers questions about videos from a Mediasite Channel video library
- Searches across transcripts, on screen text, notes, documents, and slides
- Generates summaries and key takeaways
- Creates quizzes for active recall
- Builds study guides from the available course content
- Links answers back to video timestamps when transcript timing is available

## Source Policy

AthenAI only uses:

- Mediasite video transcripts
- Study materials uploaded by the student
- Extracted text from supported files, including PDFs, Word documents, and PowerPoint presentations

AthenAI does not use:

- Internet search results
- External articles or websites
- Unprovided reference materials

This keeps answers focused on the material students are actually responsible for studying. When AthenAI adds brief
background knowledge to explain a concept or flag a likely error, it labels it as outside your material and never
cites it as a source. Deployments can turn this off with `ATHENAI_GENERAL_KNOWLEDGE=0`.

## How AthenAI Behaves

- Answers directly, then explains at the depth you need, and follows requests for shorter, simpler, or deeper answers.
- Corrects mistaken premises kindly, admits its own mistakes when shown evidence, and keeps a well-supported answer
  when someone simply insists otherwise.
- Separates what your material says, what a lecturer argues, and what is uncertain, and says when the material doesn't
  cover something.
- Helps just as fully with terse, informal, or frustrated messages. Sensitive subjects such as history of
  discrimination, sexual health, and trauma are treated as legitimate study topics.
- Declines briefly, with an alternative, only when a request would facilitate harm.
- Is an AI, not a person, and can make mistakes.

## Intended Use

AthenAI is meant to support studying, review, and comprehension. Students should verify important answers against their course materials, especially when preparing for exams or assignments.
