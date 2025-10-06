FILE_ID = 1ghNXFnwYccs0Zd0S67ksNPvxWNO3ylJz
FILE_NAME = 20250319_165700.mp4
DATA_DIR = ./data
VIDEO_PATH = $(DATA_DIR)/$(FILE_NAME)
SCRIPT_PATH = scripts/process_video.py

TEST_SECONDS ?=           # optional argument, e.g. make run TEST_SECONDS=5

all: prepare download verify run

prepare:
	@mkdir -p $(DATA_DIR)
	@echo "Data directory ready: $(DATA_DIR)"


download:
	@echo "Downloading video via gdown..."
	@gdown --fuzzy "https://drive.google.com/file/d/$(FILE_ID)/view?usp=sharing" -O $(VIDEO_PATH)
	@echo "Download completed: $(VIDEO_PATH)"

verify:
	@if [ ! -s "$(VIDEO_PATH)" ]; then \
		echo "ERROR: $(VIDEO_PATH) is empty or missing!"; \
		exit 1; \
	else \
		echo "🎬 File verified:"; \
		ls -lh $(VIDEO_PATH); \
	fi


run:
	@echo "Running pruning script..."
	@if [ -z "$(TEST_SECONDS)" ]; then \
		python $(SCRIPT_PATH) --video $(VIDEO_PATH); \
	else \
		python $(SCRIPT_PATH) --video $(VIDEO_PATH) --test-seconds $(TEST_SECONDS); \
	fi


clean:
	@echo "Cleaning up downloaded data..."
	@rm -f $(VIDEO_PATH)
	@echo "Done."


reset:
	@echo "Removing all data and outputs..."
	@rm -rf $(DATA_DIR) output
	@echo "All cleaned up."
