import os
import re
import json
from pathlib import Path

class AgentToolScorer:
    def __init__(self):
        # 定义所有标准工具列表
        self.standard_tools = [
            "get_transaction_data",
            "KYC_Verification_API", 
            "AML_Sanctions_Screening_API",
            "CTR_Evaluation_API",
            "Currency_Validation_API",
            "Fraud_Detection_Scoring_API",
            "Supervisor_Approval_Routing_API",
            "Term_Deposit_Processing_API",
            "Compliance_Aggregation_API",
            "Account_Ledger_Posting_API",
            "Receipt_Generation_API",
            "Audit_Trail_Recording_API",
        ]
        
        # 基准文件路径
        self.benchmark_file = "/Deposit/Deepseek-V3.2_graph_and_random_and_expand_affairs/standard_benchmark/benchmark_test.out"
        
        # 处理目录
        self.process_dir = "/Deposit/Deepseek-V3.2_graph_and_random_and_expand_affairs/demo/agentic"
        
    def extract_tool_calls(self, file_path):
        """从txt文件中提取所有工具调用"""
        tool_calls = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # 使用正则表达式匹配 Invoking: `工具名` 格式
                pattern = r"Invoking:\s*`([^`]+)`"
                matches = re.findall(pattern, content)
                tool_calls = matches
        except Exception as e:
            print(f"读取文件 {file_path} 时出错: {e}")
        return tool_calls
    
    def calculate_api_metrics(self, tool_calls):
        """计算API调用的三个指标：精确率、召回率、F1"""
        called_tools = set(tool_calls)
        standard_tools_set = set(self.standard_tools)
        
        # 计算真正例、假正例、假反例
        true_positives = called_tools & standard_tools_set  # 调用了且是标准工具
        false_positives = called_tools - standard_tools_set  # 调用了但不是标准工具
        false_negatives = standard_tools_set - called_tools  # 是标准工具但没调用
        
        # 计算精确率、召回率、F1
        precision = len(true_positives) / len(called_tools) if len(called_tools) > 0 else 0
        recall = len(true_positives) / len(standard_tools_set) if len(standard_tools_set) > 0 else 0
        
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "true_positives": list(true_positives),
            "false_positives": list(false_positives),
            "false_negatives": list(false_negatives),
            "coverage_ratio": f"{len(true_positives)}/{len(standard_tools_set)}"
        }
    
    def extract_prequalification_decision_from_txt(self, file_path):
        """从txt文件中提取funds_availability_verified字段的值"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # 多种可能的格式模式
                patterns = [
                    # 单引号字符串格式
                    r"'denomination_reconciliation_status':\s*'([^']+)'",
                    # 双引号字符串格式  
                    r'"denomination_reconciliation_status":\s*"([^"]+)"',
                    # 布尔值格式
                    r"'denomination_reconciliation_status':\s*(True|False|true|false)",
                    r'"denomination_reconciliation_status":\s*(True|False|true|false)',
                    # 无引号格式
                    r"'denomination_reconciliation_status':\s*([^,\s}]+)",
                    r'"denomination_reconciliation_status":\s*([^,\s}]+)',
                    # 冒号后面直接跟值的格式
                    r"denomination_reconciliation_status:\s*([^\n,}]+)",
                    # 等号格式
                    r"denomination_reconciliation_status\s*=\s*([^\n,}]+)",
                ]
                
                for pattern in patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        # 返回最后一个匹配项，并清理空白字符
                        return matches[-1].strip()
                        
        except Exception as e:
            print(f"从 {file_path} 提取funds_availability_verified时出错: {e}")
        return None
    
    def extract_final_decision_from_txt(self, file_path):
        """从txt文件中按顺序提取多个字段并拼接成字符串"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # # 新增：检测是否存在Agent执行失败字符串
                # if "Agent execution failed:" in content:
                #     return "execution_failed"  # 标记执行失败
                
                # 判断是否出现 ManageComplianceAdjudication
                has_manage_compliance = "ManageComplianceAdjudication" in content                
                keys = [
                    "compliance_check_passed",
                    "transaction_approved",
                    "approval_rejection_reason"
                ]
                
                combined_result = ""
                
                for key in keys:
                    value = ""
                    # 如果是 auto_approval_achieved 且没有出现 ManageComplianceAdjudication
                    if key == "auto_approval_achieved" and not has_manage_compliance:
                        # 直接置为空，不进行正则匹配
                        combined_result += ""
                        continue                    
                    # 更严格的 pattern 顺序
                    patterns = [
                        # 1) 布尔值 — True / False
                        fr"{key}\s*:\s*(True|False|true|false)",
                        fr"'{key}'\s*:\s*(True|False|true|false)",
                        fr'"{key}"\s*:\s*(True|False|true|false)',
                        fr"{key}\s*=\s*(True|False|true|false)",

                        # 2) 数字 — 整数 / 带小数
                        fr"{key}\s*:\s*([-+]?\d+(\.\d+)?)",
                        fr"'{key}'\s*:\s*([-+]?\d+(\.\d+)?)",
                        fr'"{key}"\s*:\s*([-+]?\d+(\.\d+)?)',
                        fr"{key}\s*=\s*([-+]?\d+(\.\d+)?)",

                        # 3) 引号字符串
                        fr"'{key}'\s*:\s*'([^']*)'",
                        fr'"{key}"\s*:\s*"([^"]*)"',
                        
                        # 4) 宽泛兜底匹配
                        fr"{key}\s*:\s*([^\n,}}]+)",
                        fr"{key}\s*=\s*([^\n,}}]+)",
                        fr"'{key}'\s*:\s*([^\s,}}]+)",
                        fr'"{key}"\s*:\s*([^\s,}}]+)',
                    ]
                    
                    for pattern in patterns:
                        matches = re.findall(pattern, content, re.IGNORECASE)
                        if matches:
                            # 有的 pattern 有多个 group，如数字中包含分组
                            raw = matches[-1]
                            if isinstance(raw, tuple):
                                # 取第一个非空的 group
                                for part in raw:
                                    if part and not part.isspace():
                                        value = part.strip()
                                        break
                            else:
                                value = raw.strip()
                            break
                    
                    combined_result += value
                
                return combined_result
        
        except Exception as e:
            print(f"从 {file_path} 提取数据时出错: {e}")
            return None
    
    def extract_benchmark_field(self, application_id, field_name):
        """从基准文件中提取对应application_id的特定字段值"""
        try:
            with open(self.benchmark_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # 查找该application_id所在的区块
                pattern = rf'transaction_id:{re.escape(application_id)}\n(.*?)(?=\napplication_id:|\Z)'
                match = re.search(pattern, content, re.DOTALL)
                
                if match:
                    # 提取该区块的内容
                    block_content = match.group(1)
                    
                    # 在区块中查找指定字段
                    field_pattern = rf'{field_name}:([^\n]+)'
                    field_match = re.search(field_pattern, block_content)
                    
                    if field_match:
                        return field_match.group(1).strip()
                        
        except Exception as e:
            print(f"从基准文件提取{field_name}时出错: {e}")
        return None
    
    def calculate_prequalification_correctness(self, application_id, txt_file_path):
        """计算预审决策正确与否的布尔值"""
        # 从txt文件提取prequalification_decision
        txt_decision = self.extract_prequalification_decision_from_txt(txt_file_path)
        
        # 从基准文件提取prequalification_decision
        benchmark_decision = self.extract_benchmark_field(application_id, "denomination_reconciliation_status")
        
        print(f"应用ID: {application_id}")
        print(f"TXT文件预审决策: '{txt_decision}'")
        print(f"基准文件预审决策: '{benchmark_decision}'")
        
        # 如果两者都存在且相同，则视为正确
        if txt_decision and benchmark_decision and txt_decision.strip() == benchmark_decision.strip():
            print(f"✓ 预审决策匹配正确")
            return True
        else:
            print(f"✗ 预审决策不匹配或缺失")
            return False
    
    def calculate_final_decision_correctness(self, application_id, txt_file_path):
        """计算终审决策正确与否的布尔值"""
        # 新增：先检测文件中是否存在Agent执行失败字符串
        try:
            with open(txt_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # if "Agent execution failed:" in content:
                #     print(f"应用ID: {application_id}")
                #     print(f"⚠️  检测到Agent执行失败，强制判定终审不通过")
                #     return False
        except Exception as e:
            print(f"检测Agent执行失败状态时出错: {e}")
        
        # 原有逻辑：从txt文件提取final_decision
        txt_decision = self.extract_final_decision_from_txt(txt_file_path)
        
        # 从基准文件提取final_decision
        benchmark_decision = self.extract_benchmark_field(application_id, "final_decision")
        
        print(f"应用ID: {application_id}")
        print(f"TXT文件终审决策: '{txt_decision}'")
        print(f"基准文件终审决策: '{benchmark_decision}'")
        
        # 如果两者都存在且相同，则视为正确
        if txt_decision and benchmark_decision and txt_decision.strip() == benchmark_decision.strip():
            print(f"✓ 终审决策匹配正确")
            return True
        else:
            print(f"✗ 终审决策不匹配或缺失")
            return False
    
    def process_single_file(self, file_path):
        """处理单个文件并计算评分"""
        file_name = os.path.basename(file_path)
        print(f"正在处理文件: {file_name}")
        
        # 提取application_id
        application_id_match = re.match(r'(TXN_DEP_[\w-]+)\.txt', file_name)
        if not application_id_match:
            print(f"文件名格式不正确: {file_name}")
            return None
        
        application_id = application_id_match.group(1)
        
        # 提取工具调用
        tool_calls = self.extract_tool_calls(file_path)
        print(f"检测到的工具调用: {len(tool_calls)} 个")
        
        # 计算三个核心指标
        api_metrics = self.calculate_api_metrics(tool_calls)
        
        # 计算预审决策正确性
        prequal_correctness = self.calculate_prequalification_correctness(application_id, file_path)
        
        # 计算终审决策正确性
        final_decision_correctness = self.calculate_final_decision_correctness(application_id, file_path)
        
        # 构建结果
        result = {
            "application_id": application_id,
            "file_name": file_name,
            "precision": api_metrics["precision"],
            "recall": api_metrics["recall"], 
            "f1_score": api_metrics["f1_score"],
            "prequalification_correctness": prequal_correctness,
            "final_decision_correctness": final_decision_correctness,
            "tool_calls_count": len(tool_calls),
            "covered_tools_count": len(api_metrics["true_positives"]),
            "true_positives": api_metrics["true_positives"],
            "false_positives": api_metrics["false_positives"],
            "false_negatives": api_metrics["false_negatives"],
            "coverage_ratio": api_metrics["coverage_ratio"],
            "txt_prequalification_decision": self.extract_prequalification_decision_from_txt(file_path),
            "benchmark_prequalification_decision": self.extract_benchmark_field(application_id, "denomination_reconciliation_status"),
            "txt_final_decision": self.extract_final_decision_from_txt(file_path),
            "benchmark_final_decision": self.extract_benchmark_field(application_id, "final_decision")
        }
        
        return result
    
    def save_result(self, result, output_dir):
        """保存结果到单独的文件"""
        if not result:
            return
        
        app_id = result["application_id"]
        output_file = os.path.join(output_dir, f"{app_id}_result.txt")
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"=== Agent工具调用分析结果 ===\n\n")
                f.write(f"申请ID: {result['application_id']}\n")
                f.write(f"文件名: {result['file_name']}\n")
                f.write(f"精确率: {result['precision']:.4f}\n")
                f.write(f"召回率: {result['recall']:.4f}\n")
                f.write(f"F1分数: {result['f1_score']:.4f}\n")
                f.write(f"预审决策正确性: {result['prequalification_correctness']}\n")
                f.write(f"终审决策正确性: {result['final_decision_correctness']}\n")
                f.write(f"TXT文件预审决策: {result['txt_prequalification_decision']}\n")
                f.write(f"基准文件预审决策: {result['benchmark_prequalification_decision']}\n")
                f.write(f"TXT文件终审决策: {result['txt_final_decision']}\n")
                f.write(f"基准文件终审决策: {result['benchmark_final_decision']}\n\n")
                
                f.write(f"工具调用统计:\n")
                f.write(f"  总调用次数: {result['tool_calls_count']}\n")
                f.write(f"  覆盖工具数: {result['covered_tools_count']} ({result['coverage_ratio']})\n")
                f.write(f"  真正例(正确调用): {', '.join(result['true_positives'][:10])}")
                if len(result['true_positives']) > 10:
                    f.write(f" ... 等 {len(result['true_positives'])} 个")
                f.write("\n")
                
                if result['false_positives']:
                    f.write(f"  假正例(错误调用): {', '.join(result['false_positives'][:10])}")
                    if len(result['false_positives']) > 10:
                        f.write(f" ... 等 {len(result['false_positives'])} 个")
                    f.write("\n")
                else:
                    f.write(f"  假正例(错误调用): 无\n")
                    
                if result['false_negatives']:
                    f.write(f"  假反例(遗漏调用): {', '.join(result['false_negatives'][:10])}")
                    if len(result['false_negatives']) > 10:
                        f.write(f" ... 等 {len(result['false_negatives'])} 个")
                    f.write("\n")
                else:
                    f.write(f"  假反例(遗漏调用): 无\n")
            
            print(f"分析结果已保存: {output_file}")
            
        except Exception as e:
            print(f"保存分析结果时出错: {e}")
    
    def process_all_files(self):
        """处理目录下的所有txt文件"""
        # 创建输出目录
        output_dir = os.path.join(self.process_dir, "analysis_results")
        os.makedirs(output_dir, exist_ok=True)
        
        # 查找所有MLN-*.txt文件
        pattern = "TXN_DEP_*.txt"
        txt_files = list(Path(self.process_dir).glob(pattern))
        
        print(f"找到 {len(txt_files)} 个待处理文件")
        
        all_results = []
        
        for file_path in txt_files:
            result = self.process_single_file(file_path)
            if result:
                self.save_result(result, output_dir)
                all_results.append(result)
                
                print(f"文件 {file_path.name} 分析完成: "
                      f"精确率={result['precision']:.4f}, "
                      f"召回率={result['recall']:.4f}, "
                      f"F1={result['f1_score']:.4f}, "
                      f"预审决策正确性={result['prequalification_correctness']}, "
                      f"终审决策正确性={result['final_decision_correctness']}")
                print("-" * 50)
        
        # 生成汇总报告
        if all_results:
            self.generate_summary_report(all_results, output_dir)
    
    def generate_summary_report(self, all_results, output_dir):
        """生成汇总报告"""
        summary_file = os.path.join(output_dir, "analysis_summary.txt")
        
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("=== Agent工具调用分析汇总报告 ===\n\n")
                f.write(f"总文件数: {len(all_results)}\n")
                
                # 计算平均指标
                avg_precision = sum(result['precision'] for result in all_results) / len(all_results)
                avg_recall = sum(result['recall'] for result in all_results) / len(all_results)
                avg_f1 = sum(result['f1_score'] for result in all_results) / len(all_results)
                
                f.write(f"平均精确率: {avg_precision:.4f}\n")
                f.write(f"平均召回率: {avg_recall:.4f}\n")
                f.write(f"平均F1分数: {avg_f1:.4f}\n")
                
                # 计算预审决策正确率
                prequal_correct_count = sum(1 for result in all_results if result['prequalification_correctness'])
                prequal_correctness_rate = prequal_correct_count / len(all_results) if all_results else 0
                
                # 计算终审决策正确率
                final_correct_count = sum(1 for result in all_results if result['final_decision_correctness'])
                final_correctness_rate = final_correct_count / len(all_results) if all_results else 0
                f.write(f"终审决策正确率: {final_correctness_rate:.4f} ({final_correct_count}/{len(all_results)})\n\n")
                
                f.write("详细分析结果:\n")
                for result in all_results:
                    f.write(f"{result['application_id']}: "
                           f"精确率={result['precision']:.4f}, "
                           f"召回率={result['recall']:.4f}, "
                           f"F1={result['f1_score']:.4f}, "
                           f"预审决策正确性={result['prequalification_correctness']} "
                           f"(TXT:{result['txt_prequalification_decision'] or '无'} vs "
                           f"基准:{result['benchmark_prequalification_decision'] or '无'}), "
                           f"终审决策正确性={result['final_decision_correctness']} "
                           f"(TXT:{result['txt_final_decision'] or '无'} vs "
                           f"基准:{result['benchmark_final_decision'] or '无'})\n")
            
            print(f"汇总报告已生成: {summary_file}")
            
        except Exception as e:
            print(f"生成汇总报告时出错: {e}")

def main():
    scorer = AgentToolScorer()
    scorer.process_all_files()

if __name__ == "__main__":
    main()