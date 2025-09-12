SHELL := /bin/bash

TEMPLATE  = template.cpp
SUBMIT    = submit.cpp
GENERATOR = generator.cpp
INPUT     = input.txt

BUILD_DIR   := build
BIN_DEBUG   := $(BUILD_DIR)/app
BIN_RELEASE := $(BUILD_DIR)/app.release
GENBIN      := $(BUILD_DIR)/generator

CXX   := ccache g++
LDOPT := $(shell command -v mold >/dev/null 2>&1 && echo -fuse-ld=mold || (command -v ld.lld >/dev/null 2>&1 && echo -fuse-ld=lld))

SAN    ?= asan
STRICT ?= 0
PCH    ?= 0

IS_CLANG := $(shell $(CXX) --version 2>/dev/null | head -n1 | grep -qi clang && echo 1 || echo 0)
ifeq ($(IS_CLANG),1)
  SHADOW_FLAGS := -Wno-shadow
else
  SHADOW_FLAGS := -Wno-shadow
endif
CXXWARN_BASE := -Wall -Wextra $(SHADOW_FLAGS) -Wno-pedantic -Wno-variadic-macros
ifeq ($(STRICT),1)
  CXXWARN := $(CXXWARN_BASE) -Werror=shadow
else
  CXXWARN := $(CXXWARN_BASE)
endif

ifeq ($(SAN),asan)
  SANFLAGS := -fsanitize=address
  STL_DEBUG_FLAGS :=
  UNDEF_STL_DEBUG := -U_GLIBCXX_DEBUG -U_GLIBCXX_ASSERTIONS
  ASAN_SO := $(shell $(CXX) -print-file-name=libasan.so)
  ifeq ($(ASAN_SO),libasan.so)
    ASAN_SO :=
  endif
else
  SANFLAGS :=
  STL_DEBUG_FLAGS := -D_GLIBCXX_DEBUG -D_GLIBCXX_ASSERTIONS
  UNDEF_STL_DEBUG :=
  ASAN_SO :=
endif

NO_PIE_CXX := -fno-pie
NO_PIE_LD  := -no-pie

CXXCOMMON = -fdiagnostics-color=auto -Wno-unused-parameter -Wno-unused-variable -Wno-unused-const-variable $(CXXWARN)
CXXDEBUG  = $(CXXCOMMON) -std=gnu++23 -O1 -pipe -DLOCAL -DDEBUG_AUTO_FLUSH $(STL_DEBUG_FLAGS) $(UNDEF_STL_DEBUG) -fno-omit-frame-pointer $(NO_PIE_CXX) $(SANFLAGS) -g
CXXREL    = $(CXXCOMMON) -std=gnu++23 -O2 -pipe -DNDEBUG $(NO_PIE_CXX)
LDDEBUG   = $(LDOPT) $(NO_PIE_LD) $(SANFLAGS)
LDREL     = $(LDOPT) $(NO_PIE_LD)

ifeq ($(PCH),1)
  PCHFILE  := $(BUILD_DIR)/.pch.$(SAN).hpp
  PCHGCH   := $(PCHFILE).gch
  USE_PCH  := -include $(PCHFILE)
  DEPS_PCH := $(PCHGCH)
else
  USE_PCH  :=
  DEPS_PCH :=
endif

.PHONY: new run run-interactive release run-release clean

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

ifeq ($(PCH),1)
$(PCHFILE): | $(BUILD_DIR)
	@printf '#include <bits/stdc++.h>\n' > '$@'
$(PCHGCH): $(PCHFILE) Makefile | $(BUILD_DIR)
	@$(CXX) $(CXXDEBUG) -x c++-header '$<' -o '$@'
endif

$(BIN_DEBUG): $(SUBMIT) $(DEPS_PCH) | $(BUILD_DIR)
	@$(CXX) $(CXXDEBUG) $(USE_PCH) -o '$@' '$(SUBMIT)' $(LDDEBUG)

$(BIN_RELEASE): $(SUBMIT) $(DEPS_PCH) | $(BUILD_DIR)
	@$(CXX) $(CXXREL)  $(USE_PCH) -o '$@' '$(SUBMIT)' $(LDREL)

$(GENBIN): $(GENERATOR) $(DEPS_PCH) | $(BUILD_DIR)
	@$(CXX) $(CXXDEBUG) $(USE_PCH) -o '$@' '$(GENERATOR)' $(LDDEBUG)

run: $(BIN_DEBUG)
	@OUT="$(BUILD_DIR)/.asan.out"; rm -f "$$OUT"; \
	LDPRE=""; [ -n '$(ASAN_SO)' ] && LDPRE="LD_PRELOAD=$(ASAN_SO)"; \
	ASANOPTS="ASAN_OPTIONS=halt_on_error=1:detect_container_overflow=1:symbolize=0:fast_unwind_on_malloc=0"; \
	stdbuf -oL -eL env $$LDPRE $$ASANOPTS '$(BIN_DEBUG)' < '$(INPUT)' > "$$OUT" 2>&1 || true; \
	if grep -q 'ERROR: AddressSanitizer' "$$OUT"; then \
	  ADDR=$$(sed -nE 's/^#0[[:space:]]+0x([0-9a-f]+).*/0x\1/p' "$$OUT" | head -n1); \
	  [ -z "$$ADDR" ] && ADDR=$$(sed -nE 's/.*pc (0x[0-9a-f]+).*/\1/p' "$$OUT" | head -n1); \
	  [ -z "$$ADDR" ] && ADDR=$$(grep -Eom1 '0x[0-9a-f]+' "$$OUT"); \
	  if [ -n "$$ADDR" ] && command -v addr2line >/dev/null 2>&1; then \
	    LINE=$$(addr2line -e '$(BIN_DEBUG)' -f -p "$$ADDR" | sed -n 's/.*:\([0-9][0-9]*\)$$/\1/p' | head -n1); \
	    if [ -n "$$LINE" ] && [ "$$LINE" != "0" ]; then echo "out_of_bound on line : $$LINE"; exit 1; fi; \
	  fi; \
	  LN=$$(grep -m1 -Eo '([[:alnum:]_./-]+\.cpp:[0-9]+)' "$$OUT" | sed -n 's/.*:\([0-9]\+\)$$/\1/p'); \
	  if [ -n "$$LN" ] && [ "$$LN" != "0" ]; then echo "out_of_bound on line : $$LN"; else echo "out_of_bound"; fi; \
	  exit 1; \
	else \
	  cat "$$OUT"; \
	fi

run-interactive: $(BIN_DEBUG)
	@'$(BIN_DEBUG)'

release: $(BIN_RELEASE)

run-release: release
	@'$(BIN_RELEASE)' < '$(INPUT)'

new: | $(BUILD_DIR)
	@cp -f '$(TEMPLATE)' '$(SUBMIT)' && > '$(INPUT)'

clean:
	@rm -rf '$(BUILD_DIR)'

