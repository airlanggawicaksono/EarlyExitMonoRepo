# Multi-tenant data transfer, keyed by run tag.
#
#   On the Jetson (inside the early-experiment session, which owns the repo):
#       make tar TAG=bert2
#
#   On the local box (Windows):
#       make get-data TAG=bert2
#
# The tarball is named after the tag, so different runs do not overwrite each
# other. Override any path on the command line:
#       make get-data TAG=bert2 JETSON=user@host DEST=somewhere

TAG     ?= run
REPO    ?= /home/earlyexp/EarlyExitMonoRepo
JETSON  ?= jetson-orin-ugm@100.114.77.15
DEST    ?= multitenant_result
TARBALL  = mt_$(TAG).tgz

.PHONY: tar get-data

## Jetson: pack this tag's data into /tmp/mt_<TAG>.tgz.
## Run inside the early-experiment session (it owns the repo, so no sudo needed).
## Includes every per-mode result CSV plus only the raw log folders whose name
## contains the tag, so one run's data is grabbed without dragging in the others.
tar:
	cd $(REPO) && tar --warning=no-file-changed -czf /tmp/$(TARBALL) \
	    result/multitenant \
	    $$(find logs -maxdepth 2 -type d -name "mt_*$(TAG)*" 2>/dev/null) \
	  && chmod 644 /tmp/$(TARBALL) && du -h /tmp/$(TARBALL)
	@echo "packed /tmp/$(TARBALL) : on the local box run  make get-data TAG=$(TAG)"

## Windows: pull /tmp/mt_<TAG>.tgz from the Jetson and extract it into DEST.
get-data:
	-mkdir $(DEST)
	scp $(JETSON):/tmp/$(TARBALL) $(DEST)/$(TARBALL)
	cd $(DEST) && tar xzf $(TARBALL)
	@echo "extracted into $(DEST)/ (tag $(TAG))"
