APP_DIR = $(abspath ./)
.IGNORE:
# Rules
all:
	@echo "App dir is at " $(APP_DIR)
	@cd $(ARCANE_LIB_PATH)/sw/applications && mkdir match_app
	@mv $(ARCANE_LIB_PATH)/sw/external/lib/drivers/match/runtime $(ARCANE_LIB_PATH)/sw/applications/match_app/runtime
	@rm -rf $(ARCANE_LIB_PATH)/sw/external/lib/drivers/match/*
	@mv $(ARCANE_LIB_PATH)/sw/applications/match_app/runtime $(ARCANE_LIB_PATH)/sw/external/lib/drivers/match/runtime
	@cp $(APP_DIR)/src/* $(ARCANE_LIB_PATH)/sw/external/lib/drivers/match -R
	@cp $(APP_DIR)/include $(ARCANE_LIB_PATH)/sw/external/lib/drivers/match -R
	@cp $(APP_DIR)/codegen $(ARCANE_LIB_PATH)/sw/external/lib/drivers/match -R
	@mv $(ARCANE_LIB_PATH)/sw/external/lib/drivers/match/main.c $(ARCANE_LIB_PATH)/sw/applications/match_app/main.c
	@cd $(ARCANE_LIB_PATH) && make app PROJECT=match_app
    
.PHONY: all
