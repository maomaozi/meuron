#include "stdafx.h"
#include "utils.h"

std::string vecShape2Str(const std::vector<int>& vec) {
	if (vec.empty()) {
		return "(NULL)";
	}

	std::string res = "(";
	char tmp[10];
	for (size_t i = 0; i < vec.size(); ++i) {
		itoa(vec[i], tmp, 10);
		res.append(tmp);
		if (i != vec.size() - 1) {
			res.append(", ");
		}
	}
	res.append(")");

	return res;
}

template<typename U, typename V>
bool checkTypeSame() {
	return std::is_same<U, V>::value;
}

bool checkShapeSame(const std::vector<int>& lhShape, const std::vector<int>& rhShape) {
	/// ������������Ƿ���״��ͬ���ҷǿ�

	/// �жϾ����Ƿ�Ϊ��
	if (!lhShape.size() || !rhShape.size()) {
		return false;
	}

	/// ���߲�������״��ͬ
	if (lhShape != rhShape) {
		return false;
	}

	return true;
}

size_t shape2size(const std::vector<int>& shape) {
	size_t dataSizeAccu = 1;

	for (const int &i : shape) {
		if (i <= 0)
		{
			return -1;
		}
		dataSizeAccu *= i;
	}

	return dataSizeAccu;
}