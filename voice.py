"""Talk to the system: ask where things are and hear the answer.

    python voice.py              type a question, hear and read the answer
    python voice.py --listen     speak the question into the microphone instead
    python voice.py --quiet      print only, no speech
    python voice.py --full       speak the whole answer instead of one short line

By default only one short line is spoken (the object and its distance); the full answer is printed.

Say or type, for example: "take me to the laptop". Say or type "quit" to stop.
Run live_run.py in another terminal so there is a position to guide from.
"""
import argparse

from guidance import describe
from sound import listen_once, speak_text

parser = argparse.ArgumentParser()
parser.add_argument('--listen', action='store_true', help='use the microphone for questions')
parser.add_argument('--quiet', action='store_true', help='do not speak answers')
parser.add_argument('--full', action='store_true', help='speak the whole answer, not one short line')
args = parser.parse_args()

print('Ask where something is. Say or type "quit" to stop.')
while True:
    if args.listen:
        print('Listening...')
        query = listen_once()
        if not query:
            print("I didn't hear anything. Try again, or run without --listen to type.")
            continue
        print(f'You said: {query}')
    else:
        try:
            query = input('> ').strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query:
            continue
    if query.lower().strip('.! ') in ('quit', 'exit', 'stop'):
        break
    answer = describe(query)
    print(answer['text'])
    print()
    if not args.quiet:
        try:                                      # speech has a time limit; Ctrl+C skips it
            speak_text(answer['spoken_full'] if args.full else answer['spoken'])
        except KeyboardInterrupt:
            print('(speech skipped)')
