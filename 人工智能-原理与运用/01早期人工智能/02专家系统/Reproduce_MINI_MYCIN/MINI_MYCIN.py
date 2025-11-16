# 知识库
class Rule:
    def __init__(self, conditions:list[str], diagnosis:str, cf:float) -> None:
        self.conditions = conditions
        self.diagnosis = diagnosis
        self.cf = cf


class KnowledgeBase:
    def __init__(self) -> None:
        self.rules = []
        self.medications_recommendations = {}
        self.define_rules()

    def add_rule(self, rule:Rule) -> None:
        self.rules.append(rule)

    def define_rules(self) -> None:
        # 定义并添加规则
        rules = [
            Rule(["温度高", "咳嗽"], "流感", 0.8),
            Rule(["温度高", "头痛"], "流感", 0.6),
            Rule(["喉咙痛", "咳嗽"], "感冒", 0.7),
            Rule(["头痛", "恶心"], "偏头痛", 0.9),
            Rule(["皮疹", "发痒"], "过敏", 0.7),
            Rule(["流鼻涕", "打喷嚏"], "过敏", 0.8),
            Rule(["发烧", "疲劳"], "病毒感染", 0.75),
            Rule(["疲劳", "肌肉酸痛"], "流感", 0.7),
            Rule(["温度高", "发烧"], "病毒感染", 0.85),
            Rule(["流鼻涕", "咳嗽"], "感冒", 0.65),
            Rule(["咳嗽", "恶心"], "胃炎", 0.6),
            Rule(["疲劳", "头痛"], "偏头痛", 0.8),
            Rule(["温度高", "咳嗽", "肌肉酸痛"], "流感", 0.9),
            Rule(["发烧", "头痛", "疲劳"], "病毒感染", 0.8),
        ]

        for rule in rules:
            self.add_rule(rule)

        self.medications_recommendations = {
            "流感": ("药物A", "A的作用 -> 治疗流感"),
            "感冒": ("药物B", "B的作用 -> 治疗感冒"),
            "偏头痛": ("药物C", "C的作用 -> 治疗偏头痛"),
            "过敏": ("药物D", "D的作用 -> 治疗过敏"),
            "病毒感染": ("药物E", "E的作用 -> 治疗病毒感染"),
            "胃炎": ("药物F", "F的作用 -> 治疗胃炎"),
        }


# 推理机部分
class InferenceEngine:
    def __init__(self, knowledge_base:KnowledgeBase) -> None:
        self.knowledge_base = knowledge_base
        self.explanation = []

    def diagnose(self, symptoms:list[str]) -> tuple[str|None, float]:
        combined_cf = {}
        self.explanation = []

        for rule in self.knowledge_base.rules:
            if all(symptom in symptoms for symptom in rule.conditions):
                if rule.diagnosis in combined_cf:
                    previous_cf = combined_cf[rule.diagnosis]
                    new_cf = self.combine_cf(previous_cf, rule.cf)
                    # 利用解释器生成推理内容
                    ExplanationGenerator.add_explanation(self.explanation, rule.conditions, rule.diagnosis, rule.cf,
                                                         previous_cf, new_cf)
                    combined_cf[rule.diagnosis] = new_cf
                else:
                    combined_cf[rule.diagnosis] = rule.cf
                    # 利用解释器生成推理内容
                    ExplanationGenerator.add_explanation(self.explanation, rule.conditions, rule.diagnosis, rule.cf)

        if not combined_cf:
            return None, 0

        final_diagnosis = max(combined_cf, key=combined_cf.get, default=None)
        final_cf = combined_cf.get(final_diagnosis, 0)
        return final_diagnosis, final_cf

    # 非确定知识推导
    def combine_cf(self, cf1:float, cf2:float) -> float:
        try:
            cf1 = float(cf1)
            cf2 = float(cf2)
        except (ValueError, TypeError):
            print(f"错误：置信度参数必须是数字，cf1={cf1}, cf2={cf2}")
            return 0.0  # 或返回其他默认值

        return cf1 + cf2 * (1 - abs(cf1))


# 解释器
class ExplanationGenerator:
    def __init__(self, inference_engine:InferenceEngine) -> None:
        self.inference_engine = inference_engine

    def explain_diagnosis(self) -> str:
        return "\n".join(self.inference_engine.explanation)

    def explain_medications(self, diagnosis:str, knowledge_base:KnowledgeBase) -> str:
        medication, reason = knowledge_base.medications_recommendations.get(diagnosis, ("无推荐药物", "无推荐原因"))
        return f"推荐药物：{medication} \n 用药原因：{reason}"

    @staticmethod
    def add_explanation(explanation_list:list[str], conditions:str, diagnosis:str, cf:float, previous_cf:float=None, new_cf:float=None) -> None:
        if previous_cf is not None and new_cf is not None:
            explanation_list.append(
                f"规则运用：如果{conditions}, 那么诊断为{diagnosis} [置信度={cf}]\n"
                f"合并置信度： {previous_cf} + {cf} * (1 - |{previous_cf}|) = {new_cf}"
            )
        else:
            explanation_list.append(
                f"规则运用：如果{conditions}, 那么诊断为{diagnosis} [置信度={cf}]"
            )

# 人机交互界面
def main():
    # 创建知识库实例
    knowledge_base = KnowledgeBase()
    # 创建推理机实例
    inference_engine = InferenceEngine(knowledge_base)
    # 创建解释器实例
    explanation_generator = ExplanationGenerator(inference_engine)
    # 棵树图症状列表
    possible_symptoms = [
        "温度高", "咳嗽", "头痛", "喉咙疼", "恶心", "皮疹",
        "发痒", "流鼻涕", "打喷嚏", "发烧", "疲劳", "肌肉酸痛"
    ]

    while True:
        print("\n可输入的症状：")
        print(",".join(possible_symptoms))

        # 用户输入
        input_symptoms = input("请输入症状（英文逗号分隔）：")
        symptoms = [symptom.strip() for symptom in input_symptoms.split(",")]

        # 诊断
        diagnosis, confidence = inference_engine.diagnose(symptoms)

        # 输出结果
        if diagnosis:
            print(f"\n诊断结果:{diagnosis} [置信度={confidence}]")

            while True:
                print("\n选择操作：")
                print("1. 查看诊断过程")
                print("2. 查看用药推荐原因")
                print("3. 继续诊断")
                print("4. 结束程序")

                # 用户选择操作
                choice = input("请输入选择(1/2/3/4)：").strip()
                match choice:
                    case "1":
                        print("解释过程：")
                        print(explanation_generator.explain_diagnosis())
                    case "2":
                        print(explanation_generator.explain_medications(diagnosis, knowledge_base))
                    case "3":
                        break
                    case "4":
                        print("诊断结束")
                        return
                    case _:
                        print("无效输入， 请输入1、2、3或4")

        else:
            print("\n无法根据输入症状作出反应")

if __name__ == '__main__':
    main()