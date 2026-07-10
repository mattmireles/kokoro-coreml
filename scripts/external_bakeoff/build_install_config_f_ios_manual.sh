#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
APP_BUNDLE_ID="${APP_BUNDLE_ID:-com.kokoro.externalbakeoff.ConfigFIOSRunnerManual}"
APP_NAME="${APP_NAME:-ConfigFIOSRunner}"
BUILD_ROOT="${BUILD_ROOT:-/tmp/kokoro-configf-manual-ios}"
APP_DIR="${BUILD_ROOT}/${APP_NAME}.app"
DEVICE_ID="${DEVICE_ID:-00008101-001134561A0A001E}"
SIGN_IDENTITY="${SIGN_IDENTITY:-CBCEEFFEE576E29E164B9A4DAD96C08655666D53}"
PROVISIONING_PROFILE="${PROVISIONING_PROFILE:-/tmp/kokoro-external-bakeoff/ios-runner-derived/Build/Products/Debug-iphoneos/SoniqoKokoroIOSRunner.app/embedded.mobileprovision}"
INSTALL="${INSTALL:-1}"

SDK_PATH="$(xcrun --sdk iphoneos --show-sdk-path)"
SWIFTC="$(xcrun --sdk iphoneos --find swiftc)"

rm -rf "${BUILD_ROOT}"
mkdir -p "${APP_DIR}"

cp "${REPO_ROOT}/scripts/external_bakeoff/ConfigFIOSRunner/Sources/Info.plist" "${APP_DIR}/Info.plist"
/usr/libexec/PlistBuddy -c "Set :CFBundleExecutable ${APP_NAME}" "${APP_DIR}/Info.plist"
/usr/libexec/PlistBuddy -c "Set :CFBundleIdentifier ${APP_BUNDLE_ID}" "${APP_DIR}/Info.plist"
/usr/libexec/PlistBuddy -c "Set :CFBundleName ${APP_NAME}" "${APP_DIR}/Info.plist"
/usr/libexec/PlistBuddy -c "Set :CFBundleDisplayName ${APP_NAME}" "${APP_DIR}/Info.plist" 2>/dev/null \
  || /usr/libexec/PlistBuddy -c "Add :CFBundleDisplayName string ${APP_NAME}" "${APP_DIR}/Info.plist"
/usr/libexec/PlistBuddy -c "Set :CFBundleShortVersionString 1.0" "${APP_DIR}/Info.plist" 2>/dev/null \
  || /usr/libexec/PlistBuddy -c "Add :CFBundleShortVersionString string 1.0" "${APP_DIR}/Info.plist"
/usr/libexec/PlistBuddy -c "Set :CFBundleVersion 1" "${APP_DIR}/Info.plist" 2>/dev/null \
  || /usr/libexec/PlistBuddy -c "Add :CFBundleVersion string 1" "${APP_DIR}/Info.plist"
/usr/libexec/PlistBuddy -c "Add :CFBundleSupportedPlatforms array" "${APP_DIR}/Info.plist" 2>/dev/null || true
/usr/libexec/PlistBuddy -c "Add :CFBundleSupportedPlatforms:0 string iPhoneOS" "${APP_DIR}/Info.plist" 2>/dev/null || true

swift_files=(
  "${REPO_ROOT}/scripts/external_bakeoff/ConfigFIOSRunner/Sources/ConfigFIOSRunnerApp.swift"
  "${REPO_ROOT}/swift/Sources/KokoroPipeline/AlignmentBuilder.swift"
  "${REPO_ROOT}/swift/Sources/KokoroPipeline/BucketSelector.swift"
  "${REPO_ROOT}/swift/Sources/KokoroPipeline/HarmonicSource.swift"
  "${REPO_ROOT}/swift/Sources/KokoroPipeline/KokoroPipeline.swift"
  "${REPO_ROOT}/swift/Sources/KokoroPipeline/KokoroSynthesisExecutor.swift"
  "${REPO_ROOT}/swift/Sources/KokoroPipeline/KokoroVocabulary.swift"
  "${REPO_ROOT}/swift/Sources/KokoroPipeline/MLMultiArrayHelpers.swift"
  "${REPO_ROOT}/swift/Sources/KokoroPipeline/PcmJoiner.swift"
  "${REPO_ROOT}/swift/Sources/KokoroPipeline/TensorDebugDump.swift"
  "${REPO_ROOT}/swift/Sources/KokoroPipeline/WaveformPostProcess.swift"
)

"${SWIFTC}" \
  -target arm64-apple-ios18.0 \
  -sdk "${SDK_PATH}" \
  -parse-as-library \
  -O \
  -g \
  -module-name "${APP_NAME}" \
  -emit-executable "${swift_files[@]}" \
  -o "${APP_DIR}/${APP_NAME}" \
  -Xlinker -rpath \
  -Xlinker @executable_path/Frameworks

cp "${PROVISIONING_PROFILE}" "${APP_DIR}/embedded.mobileprovision"

packages=(
  kokoro_duration_exact_t44.mlpackage
  kokoro_duration_exact_t105.mlpackage
  kokoro_duration_exact_t156.mlpackage
  kokoro_duration_exact_t219.mlpackage
  kokoro_duration_exact_t476.mlpackage
  kokoro_f0ntrain_t120.mlpackage
  kokoro_f0ntrain_t280.mlpackage
  kokoro_f0ntrain_t400.mlpackage
  kokoro_f0ntrain_t600.mlpackage
  kokoro_f0ntrain_t1200.mlpackage
  kokoro_decoder_pre_3s.mlpackage
  kokoro_decoder_pre_7s.mlpackage
  kokoro_decoder_pre_10s.mlpackage
  kokoro_decoder_pre_15s.mlpackage
  kokoro_decoder_pre_30s.mlpackage
  kokoro_decoder_har_post_3s.mlpackage
  kokoro_decoder_har_post_7s.mlpackage
  kokoro_decoder_har_post_10s.mlpackage
  kokoro_decoder_har_post_15s.mlpackage
  kokoro_decoder_har_post_30s.mlpackage
)

for package in "${packages[@]}"; do
  /usr/bin/ditto "${REPO_ROOT}/coreml/${package}" "${APP_DIR}/${package}"
done

for resource in 3s.json 7s.json 10s.json 15s.json 30s.json hnsf_weights.json; do
  cp "${REPO_ROOT}/outputs/swift_bench_inputs/${resource}" "${APP_DIR}/${resource}"
done

security cms -D -i "${APP_DIR}/embedded.mobileprovision" > "${BUILD_ROOT}/profile.plist"
app_prefix="$(/usr/libexec/PlistBuddy -c "Print :ApplicationIdentifierPrefix:0" "${BUILD_ROOT}/profile.plist")"
team_id="$(/usr/libexec/PlistBuddy -c "Print :Entitlements:com.apple.developer.team-identifier" "${BUILD_ROOT}/profile.plist")"

cat > "${BUILD_ROOT}/entitlements.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>application-identifier</key>
  <string>${app_prefix}.${APP_BUNDLE_ID}</string>
  <key>com.apple.developer.team-identifier</key>
  <string>${team_id}</string>
  <key>get-task-allow</key>
  <true/>
  <key>keychain-access-groups</key>
  <array>
    <string>${app_prefix}.*</string>
  </array>
</dict>
</plist>
PLIST

codesign --force --sign "${SIGN_IDENTITY}" --timestamp=none --entitlements "${BUILD_ROOT}/entitlements.plist" "${APP_DIR}"
codesign -vvv "${APP_DIR}"
du -sh "${APP_DIR}"

if [[ "${INSTALL}" == "1" ]]; then
  xcrun devicectl device install app --device "${DEVICE_ID}" "${APP_DIR}"
fi
