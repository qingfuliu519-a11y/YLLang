#ifndef YLLANG_DISTRIBUTED_INFO_H_
#define YLLANG_DISTRIBUTED_INFO_H_

namespace yllang {
class DistributedInfo {
 public:
  DistributedInfo(const int rank, int size) noexcept : m_rank_(rank), m_size_(size) {}

  auto GetRank() const -> int { return m_rank_; }
  auto GetSize() const -> int { return m_size_; }

  void SetRank(int rank) { m_rank_ = rank; }
  void SetSize(int size) { m_size_ = size; }

 private:
  int m_rank_;
  int m_size_;
};

inline static auto GetDistributedInfo() -> DistributedInfo & {
  static DistributedInfo instance(0, 1);  // 首次调用时初始化，C++11 起线程安全
  return instance;
}

inline auto SetDistributedInfo(int rank, int size) -> void {
  GetDistributedInfo().SetRank(rank);
  GetDistributedInfo().SetSize(size);
}

}  // namespace yllang

#endif  // YLLANG_DISTRIBUTED_INFO_H_