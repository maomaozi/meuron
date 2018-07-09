#include "stdafx.h"



inline void Graph::finalize()
{
	///Deal with 0-layer graph
	if (entryNodes.empty()) {
		return;
	}

	///首先处理图的计算层级模型

	std::unordered_set<__Node*> needToProcess;
	std::unordered_set<__Node*> alreadyProcessed;
	alreadyProcessed.insert(static_cast<__Node*>(nullptr));

	///第一层级为入口点
	nodesLevels.resize(1);
	for (std::shared_ptr<Tensor> ptr : entryNodes)
	{
		nodesLevels[0].insert(ptr->getNode());
		alreadyProcessed.insert(ptr->getNode());
		nodeLevelMap.insert(std::make_pair(ptr->getNode(), 0));
	}

	/// 从第0层开始遍历
	for (__Node* ptr : nodesLevels[0]) { 
		/// 首先获取完全无类型(不知道是什么Tensor也不知道数据类型)的Tensor
		/// 然后取出其中的_node(不知道什么类型的opNode)
		__Node* node = ptr->fetchNext();
		if(node)
			needToProcess.insert(ptr->fetchNext());
	}

	size_t curLayerCnt = 1;

	while (!needToProcess.empty())
	{
		std::unordered_set<__Node*> curLayer;

		for (auto node : needToProcess) {
			if (alreadyProcessed.find(node->fetchlhsParent()) != alreadyProcessed.end() && 
				alreadyProcessed.find(node->fetchrhsParent()) != alreadyProcessed.end()) {
				/// Parents are both in the current layer (or one of them is null)
				curLayer.insert(node);
			}
		}

		for (__Node* n : curLayer) {
			__Node* node = n->fetchNext();
			if (node)
				needToProcess.insert(node);

			needToProcess.erase(n);
			alreadyProcessed.insert(n);
			nodeLevelMap.insert(std::make_pair(n, curLayerCnt));
		}

		nodesLevels.push_back(std::move(curLayer));
		++curLayerCnt;
	}

}

void Graph::run(Tensor *tensor) 
{
	if (0) finalize();

	size_t layerNum = nodeLevelMap[tensor->getNode()];

	for (size_t i = 0; i < layerNum; ++i) {
		for (auto &node : nodesLevels[i]) {
			node->execHere();
		}
	}

	tensor->getNode()->execHere();
}
