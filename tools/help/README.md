# Help topics

`help_content.py` holds the text of the Help tab in English and Ukrainian. `python tools/help/gen_help.py` writes from it
`assets/ui/help_topics/*.xml`, `src/help/help_topic_table.h` and the topic list in `assets/ui/pets.xml`. The strings of the
topics (`help.*`) are written into `assets/locale/*.strings` by `python tools/gen_locale.py`, which holds all the other
messages of the game as well. Run both after a change of the help text.
