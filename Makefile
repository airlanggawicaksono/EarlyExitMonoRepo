# Multi-tenant data transfer and analysis, keyed by run tag.
#
# The flow is three steps, one door each:
#   On the Jetson (inside the early-experiment session, which owns the repo):
#       make tar TAG=camptest
#   On the local box (Windows):
#       make get-data TAG=camptest
#       make analyze  TAG=camptest MODE=maxn_super
#
# The tarball is named after the tag, so runs do not overwrite each other.
# Override any path on the command line:
#       make get-data TAG=camptest JETSON=user@host DEST=somewhere

# On Windows, run recipes through cmd.exe so PATH tools (scp, tar) resolve.
# On the Jetson (Linux) $(OS) is unset, so the default /bin/sh is kept and
# `make tar` works unchanged.
ifeq ($(OS),Windows_NT)
  SHELL := cmd.exe
  .SHELLFLAGS := /c
  PY := python
else
  PY := python3
endif

TAG     ?= run
REPO    ?= /home/earlyexp/EarlyExitMonoRepo
JETSON  ?= jetson-orin-ugm@100.114.77.15
DEST    ?= multitenant_result
MODE    ?= maxn_super
TARBALL  = mt_$(TAG).tgz

.PHONY: help tar get-data analyze

help:
	@echo "make tar TAG=x        (Jetson) pack raw logs + CSVs for tag x into /tmp/mt_x.tgz"
	@echo "make get-data TAG=x   (local)  pull and extract the tarball into $(DEST)/"
	@echo "make analyze TAG=x MODE=maxn_super  (local) run the raw analyzer + per-second view"

## Jetson: pack this tag's RAW data into /tmp/mt_<TAG>.tgz.
## Run inside the early-experiment session (it owns the repo, so no sudo needed).
## Grabs ONLY the raw hw_results.json files whose path contains the tag (the
## per-sample truth the analyzer recomputes from), plus the tiny
## concurrent_slowdown.csv the analyzer reads for one thing only: the intended
## tenant count that distinguishes a COMPLETE cell from an OOM-truncated one.
## Everything else in the log tree (plots, quality jsons, images) is left out.
tar:
	cd $(REPO) && tar --warning=no-file-changed -czf /tmp/$(TARBALL) \
	    $$(find logs -path "*$(TAG)*" -name hw_results.json 2>/dev/null) \
	    $$(find result/multitenant -name concurrent_slowdown.csv 2>/dev/null) \
	  && chmod 644 /tmp/$(TARBALL) && du -h /tmp/$(TARBALL)
	@echo "packed /tmp/$(TARBALL) : on the local box run  make get-data TAG=$(TAG)"

## Windows: pull /tmp/mt_<TAG>.tgz from the Jetson and extract it into DEST.
get-data:
	-mkdir $(DEST)
	scp $(JETSON):/tmp/$(TARBALL) $(DEST)/$(TARBALL)
	cd $(DEST) && tar xzf $(TARBALL)
	@echo "extracted into $(DEST)/ (tag $(TAG)); next: make analyze TAG=$(TAG) MODE=$(MODE)"

## Local: analyse the pulled raw data from hw_results.json (never the aggregate
## CSV). Writes the per-cell metrics and the per-second series as CSVs alongside.
analyze:
	$(PY) multitenant_analyze.py $(DEST)/logs/multitenant.$(MODE) \
	    --mode-label $(MODE) \
	    --out $(DEST)/analysis_$(TAG)_$(MODE).csv \
	    --timeseries --ts-out $(DEST)/timeseries_$(TAG)_$(MODE).csv
