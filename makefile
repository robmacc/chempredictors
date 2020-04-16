# -----------------------------------------------------------------------------
# sphinx commands
# ==============

init-docs:
	sphinx-quickstart docs

build-docs:
	rm -rf docs/_build
	sphinx-build -b singlehtml docs docs/_build

# -----------------------------------------------------------------------------
# conda commands
# ==============

# install environment:
	# conda env create --quiet -f environment.yml

# activate environment:
	# conda activate autoencoding

# -----------------------------------------------------------------------------
# pip and venv commands
# =====================

# new environment
# ---------------
	# python -m venv .env

# delete environment
# ---------------
	# rm -rf .env

# activate environment
# ---------------
	# source .env/Scripts/activate

# install dependencies
# --------------------
	# pip install -r requirements.txt

# freeze environment
# ------------------
	# pip freeze > requirements.txt

# install environment
# -------------------
	# python -m venv .env
	# source .env/Scripts/activate
	# pip install -r requirements.txt

# -----------------------------------------------------------------------------

# pytest commands
# ===============

# Run all tests
# -------------
	# pytest

# -----------------------------------------------------------------------------

# flake commands
# ===============
	# flake8 src tests