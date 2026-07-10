#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

static void Usage(NSString *message) {
    if (message.length > 0) {
        fprintf(stderr, "%s\n", message.UTF8String);
    }
    fprintf(stderr,
            "Usage: inspect_coreml_compute_plan --model PATH "
            "[--compute-units all|cpuAndGPU|cpuAndNeuralEngine|cpuOnly]\n");
    exit(2);
}

static MLComputeUnits ComputeUnits(NSString *raw) {
    if ([raw isEqualToString:@"all"]) return MLComputeUnitsAll;
    if ([raw isEqualToString:@"cpuAndGPU"]) return MLComputeUnitsCPUAndGPU;
    if ([raw isEqualToString:@"cpuAndNeuralEngine"]) return MLComputeUnitsCPUAndNeuralEngine;
    if ([raw isEqualToString:@"cpuOnly"]) return MLComputeUnitsCPUOnly;
    Usage([NSString stringWithFormat:@"unsupported compute units: %@", raw]);
    return MLComputeUnitsAll;
}

static NSString *DeviceName(id<MLComputeDeviceProtocol> device) {
    NSString *typeName = NSStringFromClass([device class]);
    if ([typeName containsString:@"NeuralEngine"]) return @"neuralEngine";
    if ([typeName containsString:@"GPU"]) return @"gpu";
    if ([typeName containsString:@"CPU"]) return @"cpu";
    return typeName ?: @"unknown";
}

static void Increment(NSMutableDictionary<NSString *, NSNumber *> *dict, NSString *key) {
    dict[key] = @([dict[key] integerValue] + 1);
}

static void AddDouble(NSMutableDictionary<NSString *, NSNumber *> *dict, NSString *key, double value) {
    dict[key] = @([dict[key] doubleValue] + value);
}

static NSMutableArray<MLModelStructureProgramOperation *> *CollectOperations(MLModelStructureProgramBlock *block) {
    NSMutableArray<MLModelStructureProgramOperation *> *operations = [NSMutableArray array];
    for (MLModelStructureProgramOperation *operation in block.operations) {
        [operations addObject:operation];
        for (MLModelStructureProgramBlock *nested in operation.blocks) {
            [operations addObjectsFromArray:CollectOperations(nested)];
        }
    }
    return operations;
}

static NSMutableDictionary *MutableOpSummary(NSMutableDictionary *byOp, NSString *opName) {
    NSMutableDictionary *summary = byOp[opName];
    if (!summary) {
        summary = [@{
            @"count": @0,
            @"costWeight": @0.0,
            @"preferredDevices": [NSMutableDictionary dictionary],
            @"supportedDevices": [NSMutableDictionary dictionary],
        } mutableCopy];
        byOp[opName] = summary;
    }
    return summary;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (@available(macOS 14.4, *)) {
            NSString *modelPath = nil;
            NSString *computeUnitsRaw = @"all";
            for (int i = 1; i < argc; i++) {
                NSString *arg = [NSString stringWithUTF8String:argv[i]];
                if ([arg isEqualToString:@"--model"]) {
                    if (++i >= argc) Usage(@"--model requires a path");
                    modelPath = [NSString stringWithUTF8String:argv[i]];
                } else if ([arg isEqualToString:@"--compute-units"]) {
                    if (++i >= argc) Usage(@"--compute-units requires a value");
                    computeUnitsRaw = [NSString stringWithUTF8String:argv[i]];
                } else if ([arg isEqualToString:@"--help"] || [arg isEqualToString:@"-h"]) {
                    Usage(@"");
                } else {
                    Usage([NSString stringWithFormat:@"unknown argument: %@", arg]);
                }
            }
            if (!modelPath) Usage(@"--model is required");

            NSURL *modelURL = [NSURL fileURLWithPath:modelPath];
            NSError *error = nil;
            NSURL *compiledURL = modelURL;
            if (![modelURL.pathExtension isEqualToString:@"mlmodelc"]) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
                compiledURL = [MLModel compileModelAtURL:modelURL error:&error];
#pragma clang diagnostic pop
                if (!compiledURL) {
                    fprintf(stderr, "compile failed: %s\n", error.localizedDescription.UTF8String);
                    return 1;
                }
            }

            MLModelConfiguration *configuration = [MLModelConfiguration new];
            configuration.computeUnits = ComputeUnits(computeUnitsRaw);

            dispatch_semaphore_t sema = dispatch_semaphore_create(0);
            __block MLComputePlan *plan = nil;
            __block NSError *planError = nil;
            [MLComputePlan loadContentsOfURL:compiledURL
                               configuration:configuration
                           completionHandler:^(MLComputePlan *_Nullable loaded, NSError *_Nullable loadError) {
                plan = loaded;
                planError = loadError;
                dispatch_semaphore_signal(sema);
            }];
            dispatch_semaphore_wait(sema, DISPATCH_TIME_FOREVER);
            if (!plan) {
                fprintf(stderr, "compute plan failed: %s\n", planError.localizedDescription.UTF8String);
                return 1;
            }
            MLModelStructureProgram *program = plan.modelStructure.program;
            if (!program) {
                fprintf(stderr, "model is not an ML Program\n");
                return 1;
            }
            MLModelStructureProgramFunction *mainFunction = program.functions[@"main"];
            if (!mainFunction) {
                fprintf(stderr, "ML Program has no main function\n");
                return 1;
            }

            NSArray<MLModelStructureProgramOperation *> *operations = CollectOperations(mainFunction.block);
            NSMutableDictionary<NSString *, NSNumber *> *preferredCounts = [NSMutableDictionary dictionary];
            NSMutableDictionary<NSString *, NSNumber *> *supportedCounts = [NSMutableDictionary dictionary];
            NSMutableDictionary<NSString *, NSNumber *> *costByPreferred = [NSMutableDictionary dictionary];
            NSMutableDictionary *byOp = [NSMutableDictionary dictionary];

            for (MLModelStructureProgramOperation *operation in operations) {
                NSString *opName = operation.operatorName ?: @"unknown";
                MLComputePlanCost *cost = [plan estimatedCostOfMLProgramOperation:operation];
                double weight = cost ? cost.weight : 0.0;
                MLComputePlanDeviceUsage *usage = [plan computeDeviceUsageForMLProgramOperation:operation];
                NSString *preferred = usage ? DeviceName(usage.preferredComputeDevice) : @"unknown";

                Increment(preferredCounts, preferred);
                AddDouble(costByPreferred, preferred, weight);

                NSMutableDictionary *summary = MutableOpSummary(byOp, opName);
                summary[@"count"] = @([summary[@"count"] integerValue] + 1);
                summary[@"costWeight"] = @([summary[@"costWeight"] doubleValue] + weight);
                Increment(summary[@"preferredDevices"], preferred);

                NSArray *supported = usage ? usage.supportedComputeDevices : @[];
                if (supported.count == 0) {
                    Increment(supportedCounts, @"unknown");
                    Increment(summary[@"supportedDevices"], @"unknown");
                } else {
                    NSMutableSet<NSString *> *seen = [NSMutableSet set];
                    for (id<MLComputeDeviceProtocol> device in supported) {
                        [seen addObject:DeviceName(device)];
                    }
                    for (NSString *deviceName in seen) {
                        Increment(supportedCounts, deviceName);
                        Increment(summary[@"supportedDevices"], deviceName);
                    }
                }
            }

            NSDictionary *report = @{
                @"model": modelURL.path,
                @"compiledModel": compiledURL.path,
                @"computeUnits": computeUnitsRaw,
                @"operationCount": @(operations.count),
                @"preferredDeviceCounts": preferredCounts,
                @"supportedDeviceCounts": supportedCounts,
                @"costWeightByPreferredDevice": costByPreferred,
                @"operations": byOp,
            };
            NSData *json = [NSJSONSerialization dataWithJSONObject:report
                                                           options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys
                                                             error:&error];
            if (!json) {
                fprintf(stderr, "json failed: %s\n", error.localizedDescription.UTF8String);
                return 1;
            }
            fwrite(json.bytes, 1, json.length, stdout);
            fputc('\n', stdout);
            return 0;
        } else {
            fprintf(stderr, "MLComputePlan requires macOS 14.4 or newer\n");
            return 1;
        }
    }
}
