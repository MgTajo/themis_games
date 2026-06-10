# Themis Mechanism: Cooperation Games

An interactive [Streamlit](https://streamlit.io/) app illustrating three cooperation
games from Carl Edward Rasmussen's talk *"Climate Change Cooperation: the Themis
Mechanism"*. You play one player, the other 99 are simulated agents you tune in the
sidebar.

1. **Independent commitment** — collapses into free-riding.
2. **Fixed budget** — zero-sum and adversarial.
3. **Common commitment** — makes cooperation a self-interested choice. Two settlement
   rules: the common minimum, and an optimal threshold that maximises the total raised
   (the Themis elicitation process).

## Getting started

You need Python 3.10+.

```bash
# 1. (optional) create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3. run the app
streamlit run themis_games.py
```

The app opens automatically in your browser.
