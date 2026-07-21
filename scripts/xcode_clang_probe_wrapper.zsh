#!/bin/zsh

# Xcode 26.6's SwiftBuild service can deadlock when the capability probe sends
# both clang's macro table and verbose driver diagnostics through one small
# pipe. The macro table is the data SwiftBuild consumes; `-v` is diagnostic
# chatter. Remove only that flag for `-E -dM` probes and preserve every real
# compiler invocation byte-for-byte.

real_clang=/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang
arguments=("$@")

if [[ " ${arguments[*]} " == *" -dM "* && " ${arguments[*]} " == *" -E "* ]]; then
  filtered_arguments=()
  for argument in "${arguments[@]}"; do
    if [[ "$argument" != "-v" ]]; then
      filtered_arguments+=("$argument")
    fi
  done
  exec "$real_clang" "${filtered_arguments[@]}"
fi

exec "$real_clang" "${arguments[@]}"
