from run_cli import run_agent


TEST_QUERIES = [
    "请帮我总结一篇多模态偏见论文",
    "请帮我分析 coco_val_n300_g1 的幻觉风险",
    "请推荐适合做幻觉评估的数据集",
    "帮我生成组会汇报文本",
    "ModuleNotFoundError: No module named langgraph 怎么解决",
    "我今天应该怎么安排科研任务",
]

'''
def main():
    for query in TEST_QUERIES:
        result = run_agent(query)
        print("=" * 60)
        print("用户输入：", query)
        print(result["final_answer"])
'''


def main():
    graph = build_graph()

    for query in TEST_QUERIES:
        result = graph.invoke(create_initial_state(query))

        print("=" * 60)
        print("用户输入：", query)
        print(result["final_answer"])
        

if __name__ == "__main__":
    main()