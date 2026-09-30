# 跨层结构剪枝预算搜索器（Java，纯标准库，Linux CPU）

为小型前馈残差网络选择“每层一个离散裁剪候选”，在**总保留参数上限**内，
通过**实际执行组合后的网络**最小化端到端校准输出误差（逐样本 MSE）。

- 只使用 Java 标准库，不下载模型、不调用任何外部服务。
- 每个方案都真正前向执行组合网络评分，绝不把各层局部误差简单相加。
- 固定种子、有限评估次数；先探索较粗裁剪等级，再在当前有效方案附近细化。
- 始终保留当前最佳可行结果；达到评估上限正常返回；无可行方案时明确报告。
- 启发式结果**不会**被标记为全局最优；小规模可用内置穷举参考给出真实最优与差距。

## 目录结构

```
verify.sh                         # 编译并实际运行自动测试（根目录）
src/main/java/pruner/
  Activation.java                 # 恒等/ReLU/Tanh/Sigmoid 激活
  LayerCandidate.java             # 一层的一个裁剪候选（已裁剪/补偿的权重、偏置、
                                  #   真实保留参数数、裁剪等级）
  LayerSpec.java                  # 一层配置：激活、可选残差来源、候选列表
  NetworkDefinition.java          # 网络与静态校验（可组合性、残差宽度、非法配置）
  ComposedNetwork.java            # 具体方案的真实端到端前向执行（含残差相加）
  CompositionException.java       # 具体选择无法组合时抛出
  CalibrationData.java            # 校准输入与参考输出
  Scorer.java                     # 端到端逐样本 MSE（非局部误差求和）
  PruningSearcher.java            # 粗到细、固定种子、有评估上限的搜索器
  SearchResult.java               # 各层选择、总参数数、误差、实际评估次数
  ExhaustiveReference.java        # 小空间穷举最优参考（测试/校验用）
  Demo.java                       # 命令行演示
src/test/java/pruner/PrunerTests.java  # 自动测试（反射运行，零用例直接失败）
```

## 构建与测试

```bash
./verify.sh
```

脚本会编译主代码与测试并实际运行。若编译失败、任一测试失败或**发现的
测试用例数为 0**，脚本以非零状态退出。

需要 JDK（在 OpenJDK 17 上验证；目标语法为 Java 8 兼容）。

手动构建：

```bash
mkdir -p build/classes
javac -d build/classes $(find src/main/java -name '*.java')
javac -cp build/classes -d build/classes $(find src/test/java -name '*.java')
java -cp build/classes pruner.PrunerTests
```

## 运行演示

```bash
java -cp build/classes pruner.Demo [预算] [种子] [评估上限]
# 例：紧评估上限（展示启发式与穷举最优的差距）
java -cp build/classes pruner.Demo 32 12345 2
# 例：宽松上限（小空间下达到穷举最优，差距 0）
java -cp build/classes pruner.Demo 32 12345 1000
# 例：极紧预算（明确报告无可行方案）
java -cp build/classes pruner.Demo 5 12345 1000
```

## 使用方式

```java
NetworkDefinition network = new NetworkDefinition(inputDim, layers);
CalibrationData data = new CalibrationData(inputs, referenceOutputs);
PruningSearcher searcher = new PruningSearcher(network, data);

SearchResult result = searcher.search(
        /* budget */ 100L,
        /* seed */ 12345L,
        /* evaluationLimit */ 200);

if (result.isFeasible()) {
    int[] perLayer = result.selection();          // 每层候选下标
    long params   = result.totalRetainedParams(); // 必然 <= budget
    double error  = result.error();               // 端到端 MSE
    int used      = result.evaluations();         // 实际评估次数
    boolean hit   = result.isEvaluationLimitReached();
} else {
    // 没有任何可组合且不超预算的方案
}
```

若还需要按顺序检查“实际评估了哪些不同方案”（重现性/额度核对），使用
`searcher.run(...)` 返回的 `SearchRun.evaluationTrace()`。

## 网络与候选约定

- 权重按 `weights[out][in]` 列主序；输出为
  `act(weights·x + biases)`。
- 层可声明残差来源：`-1` 表示网络输入，`0..i-1` 表示更早某层的激活；
  残差在**激活前相加**，且要求来源宽度等于本层输出宽度。
- 裁剪等级 `level`：越大越粗（保留越少），`0` 为未裁剪基线。搜索先在每层
  较粗的一端采样，再围绕当前最佳逐层换用相邻更细/更粗候选。
- `retainedParams` 必须与实际提供的权重+偏置元素个数一致（结构化裁剪后
  张量即“真实保留参数”），并对负值与非有限值报错。

## 评分、额度与确定性

- 一次“评估”= 成功组合并真实执行整套校准样本计算 MSE；
  维度不可组合或超预算的选择不会被评估、不扣额度。
- 同一层内若存在**张量完全相同的重复候选**，它们描述同一组合方案，
  共享缓存结果，**只扣一次**评估额度。
- 误差并列时，按各层候选下标序列的字典序选择较小者，结果稳定，与
  提案先后顺序无关。
- 搜索器无状态，相同 `(预算, 种子, 上限, 网络, 校准集)` 必然产生相同的
  评估序列与最终选择。
- 达到 `evaluationLimit` 时正常返回当前最佳；`toString()` 明确标注
  “heuristic, not claimed globally optimal”。

## 非法输入（构造期即拒绝）

空候选层、空层列表、非正输入维；候选权重非矩形（行宽不一）、空张量、
含 `NaN/Infinity`；负保留参数数、负裁剪等级、参数数与张量元素数不符；
相邻层无可组合宽度、候选无法参与任意端到端链路；残差引用非法或在任何
可组合链路上都无法宽度对齐；校准集为空、向量长度不一致、含非有限值、
与网络输入/输出维不符；负预算或负评估上限。

## 测试覆盖（19 个用例）

端到端误差（含手算 MSE 与独立残差前向交叉核对）、预算守恒（多预算多种子）、
等于预算边界、种子重现（多种子评估序列逐拍相同且不同种子路径不同）、
评估上限严格生效与上限为 0、重复候选不重复扣额度、误差并列的稳定裁决、
非法输入与不可组合配置报错、极紧预算无可行方案且不宣称全局最优。
小规模夹具另用 `ExhaustiveReference` 穷举给出最优，明确打印搜索结果与参考
的差距（紧上限差距 > 0，宽松上限差距 0）。
