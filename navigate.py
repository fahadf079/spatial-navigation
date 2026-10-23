"""Directions to a stored object, with obstacle warnings.

    python navigate.py "take me to the laptop"
    python navigate.py "take me to the laptop" --speak     also say it out loud

Matches the query against stored labels (whole phrases or single words, any case, plurals
handled), uses the newest match from the CURRENT session, and prints a turn angle and distance
relative to the camera's heading. Objects that stand near the straight path are reported as
obstacles. If the object was only seen in an older session it says so and gives no distance.
"""
import argparse

from guidance import describe
from sound import speak_text

parser = argparse.ArgumentParser()
parser.add_argument('query', nargs='+', help='for example: take me to the laptop')
parser.add_argument('--speak', action='store_true', help='also say the answer out loud')
args = parser.parse_args()

answer = describe(' '.join(args.query))
print()
print(answer['text'])
if args.speak:
    speak_text(answer['spoken'])
