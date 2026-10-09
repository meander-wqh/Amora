#ifndef PRIVATE_HNSW_V2_H
#define PRIVATE_HNSW_V2_H

#include "hnsw_quantized.h"
#include "subgroup/subgroup_manager.h"
#include "../simplepir/embedding_pir.h"
#include "../simplepir/neighbor_pir.h"
#include "../simplepir/simple_pir.h"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <memory>
#include <chrono>
#include <cmath>

namespace hnsw {

using EmbElem = simplepir::Elem32;
using NbrElem = simplepir::Elem32;

using EmbMatrix = simplepir::MatrixT<EmbElem>;
using EmbQueryMsg = simplepir::QueryMsgT<EmbElem>;
using EmbAnswerMsg = simplepir::AnswerMsgT<EmbElem>;
using EmbQueryContext = simplepir::QueryContextT<EmbElem>;
using EmbDatabase = simplepir::EmbeddingDatabaseT<EmbElem>;
using EmbPIRServer = simplepir::EmbeddingPIRServerT<EmbElem>;
using EmbPIRClient = simplepir::EmbeddingPIRClientT<EmbElem>;

using NbrMatrix = simplepir::MatrixT<NbrElem>;
using NbrQueryMsg = simplepir::NbrQueryMsgT<NbrElem>;
using NbrAnswerMsg = simplepir::NbrAnswerMsgT<NbrElem>;
using NbrQueryContext = simplepir::NbrQueryContextT<NbrElem>;
using NbrDatabase = simplepir::NeighborDatabaseT<NbrElem>;
using NbrPIRServer = simplepir::NeighborPIRServerT<NbrElem>;
using NbrPIRClient = simplepir::NeighborPIRClientT<NbrElem>;

template<typename ElemType>
struct PIRPrecomputeT {
    std::shared_ptr<simplepir::MatrixT<ElemType>> As;
    std::shared_ptr<simplepir::MatrixT<ElemType>> Hs;
};
using EmbPIRPrecompute = PIRPrecomputeT<EmbElem>;
using NbrPIRPrecompute = PIRPrecomputeT<NbrElem>;

struct PrivateHNSWConfigV2 {
    int embeddingDim = 0;
    int numNodes = 0;
    int numClusters = 0;
    int maxClusterSize = 0;
    int maxNeighbors = 0;
    uint64_t logQ = 32;
    uint64_t embeddingLogQ = 32;
    uint64_t lweN = 1024;
    bool isUnsigned = true;

    bool useSubgroupNeighborPIR = false;
    int totalSubgroups = 0;
    int maxSubgroupSize = 0;
    int targetSubgroupSize = 120;
    int maxSGPerCluster = 0;

    int partBits = 7;
    int numParts = 3;
    int localBits = 14;
    int clusterBits = 7;

    bool disableCentroidPrune = false;
    int topCand = 1;
    bool disableBatchPIR = false;
    int maxEmbRounds = 0;
    bool useCentroidEntry = true;
    int centroidEntryK = 1;
    int maxStaleRounds = 0;
    int maxCandidatesPerRound = 2;
    int maxClustersPerRound = 2;
    int fixedSearchIterations = 5;
    int maxResultsPerCluster = 20;
    void print() const;
};

struct PrivateHNSWMetadataV2 {
    int numNodes = 0;
    int numClusters = 0;
    int maxClusterSize = 0;
    int embeddingDim = 0;
    int maxNeighbors = 0;

    int entryPoint = 0;
    int entryCluster = 0;

    std::vector<std::vector<int>> clusterToNodes;
    std::vector<int> nodeToCluster;
    std::vector<int> nodeIndexInCluster;

    int maxLevel = 0;
    std::vector<int> nodeLevels;
    std::vector<std::vector<std::vector<int>>> allLevelNeighbors;
    std::vector<int> upperLayerNodes;
    std::vector<std::vector<uint8_t>> upperLayerEmbeddings;
    std::vector<int> nodeToUpperIdx;

    std::vector<std::vector<uint8_t>> clusterCentroids;

    bool hasSubgrouping = false;
    int totalSubgroups = 0;
    int maxSubgroupSize = 0;

    std::vector<std::vector<int>> clusterSubgroupOffset;

    std::vector<int> nodeToNewId;
    std::vector<int> newIdToNode;

    std::vector<int> clusterOffset;

    std::vector<int> nodeSubgroup;

    std::vector<int> nodeLocalIdxInSubgroup;

    std::vector<NodeNeighborInfo> nodeNeighborGroups;

    std::vector<std::vector<int>> subgroupToColumn;
    std::vector<int> numSubgroupsPerCluster;

    int getLocalIndexInSubgroup(int nodeId) const;
    std::pair<int, int> getNodeSubgroup(int nodeId) const;

    void print() const;
};

struct ClusterPIRResult {
    std::vector<int64_t> innerProducts;
    std::vector<std::vector<bool>> neighborClusters;
};

struct PrivateSearchStatsV2 {
    int pirQueryCount = 0;
    int embeddingPirCount = 0;
    int neighborPirCount = 0;
    int clustersAccessed = 0;
    int clustersSkipped = 0;
    int nodesVisited = 0;
    int resultsFromNeighbor = 0;
    int resultsFromCluster = 0;
    int cacheHits = 0;
    double totalPirTimeMs = 0;
    double totalSearchTimeMs = 0;

    uint64_t embQueryBytes = 0;
    uint64_t embAnswerBytes = 0;
    uint64_t nbrQueryBytes = 0;
    uint64_t nbrAnswerBytes = 0;

    uint64_t totalQueryBytes() const { return embQueryBytes + nbrQueryBytes; }
    uint64_t totalAnswerBytes() const { return embAnswerBytes + nbrAnswerBytes; }
    uint64_t totalBytes() const { return totalQueryBytes() + totalAnswerBytes(); }

    int embCommRounds = 0;
    int nbrCommRounds = 0;
    int totalCommRounds() const { return embCommRounds + nbrCommRounds; }

    double precomputeHsTimeMs = 0;
    double upperLayerSearchTimeMs = 0;
    double candidateManageTimeMs = 0;

    double embQueryGenTimeMs = 0;
    double embServerTimeMs = 0;
    double embRecoverTimeMs = 0;

    double nbrQueryGenTimeMs = 0;
    double nbrServerTimeMs = 0;
    double nbrRecoverTimeMs = 0;
    double nbrDecodeTimeMs = 0;

    double cacheWriteTimeMs = 0;

    double totalEmbPirTimeMs() const { return embQueryGenTimeMs + embServerTimeMs + embRecoverTimeMs; }
    double totalNbrPirTimeMs() const { return nbrQueryGenTimeMs + nbrServerTimeMs + nbrRecoverTimeMs + nbrDecodeTimeMs; }
    double onlineTimeMs() const {
        return totalEmbPirTimeMs() + totalNbrPirTimeMs() + upperLayerSearchTimeMs +
               candidateManageTimeMs + cacheWriteTimeMs;
    }

    void printDetailedTiming() const;
};

class PrivateHNSWServerV2 {
public:
    PrivateHNSWServerV2() = default;

    void build(HNSWQuantizedIndex& index);

    std::shared_ptr<EmbMatrix> setupEmbeddingPIR(
        const std::shared_ptr<EmbMatrix>& sharedMatrix
    );

    void setupEmbeddingPIR(
        const std::shared_ptr<EmbMatrix>& sharedMatrix,
        const std::shared_ptr<EmbMatrix>& cachedHint
    );

    std::shared_ptr<NbrMatrix> setupNeighborPIR(
        const std::shared_ptr<NbrMatrix>& sharedMatrix
    );

    void setupNeighborPIR(
        const std::shared_ptr<NbrMatrix>& sharedMatrix,
        const std::shared_ptr<NbrMatrix>& cachedHint
    );

    EmbAnswerMsg answerEmbeddingPIR(const EmbQueryMsg& query) const;

    NbrAnswerMsg answerNeighborPIR(const NbrQueryMsg& query) const;

    std::shared_ptr<EmbMatrix> batchAnswerEmbeddingPIR(
        const std::shared_ptr<EmbMatrix>& queryMatrix) const;

    std::shared_ptr<NbrMatrix> batchAnswerNeighborPIR(
        const std::shared_ptr<NbrMatrix>& queryMatrix) const;

    const PrivateHNSWConfigV2& getConfig() const { return config_; }
    const PrivateHNSWMetadataV2& getMetadata() const { return metadata_; }
    const simplepir::EmbeddingPIRParams& getEmbeddingPIRParams() const { return embPirParams_; }
    const simplepir::NeighborPIRParams& getNeighborPIRParams() const { return nbrPirParams_; }

    bool isReady() const { return isReady_; }

private:
    void buildMetadata(HNSWQuantizedIndex& index);
    void buildSubgroupNeighborPIRDatabase(const HNSWQuantizedIndex& index);

    PrivateHNSWConfigV2 config_;
    PrivateHNSWMetadataV2 metadata_;
    SubgroupManager subgroupManager_;

    simplepir::EmbeddingPIRParams embPirParams_;
    EmbDatabase embDatabase_;
    EmbPIRServer embPirServer_;

    simplepir::NeighborPIRParams nbrPirParams_;
    NbrDatabase nbrDatabase_;
    NbrPIRServer nbrPirServer_;

    bool isReady_ = false;
};

class PrivateHNSWClientV2 {
public:
    PrivateHNSWClientV2() = default;

    void init(
        const PrivateHNSWMetadataV2& metadata,
        const PrivateHNSWConfigV2& config
    );

    void initEmbeddingPIR(
        const simplepir::EmbeddingPIRParams& params,
        const std::shared_ptr<EmbMatrix>& sharedMatrix,
        const std::shared_ptr<EmbMatrix>& hint
    );

    void initNeighborPIR(
        const simplepir::NeighborPIRParams& params,
        const std::shared_ptr<NbrMatrix>& sharedMatrix,
        const std::shared_ptr<NbrMatrix>& hint
    );

    std::vector<std::pair<int, int64_t>> search(
        const uint8_t* query, int k, int ef,
        PrivateHNSWServerV2& server
    );

    std::vector<std::pair<int, int64_t>> searchWithStats(
        const uint8_t* query, int k, int ef,
        PrivateHNSWServerV2& server,
        PrivateSearchStatsV2& stats
    );

    void clearCache();
    bool isReady() const { return isReady_; }

    void precomputeSecretPool(int poolSize);

private:
    void batchQueryClusters(
        const std::vector<int>& clusterIds,
        const uint8_t* query,
        PrivateHNSWServerV2& server,
        PrivateSearchStatsV2& stats
    );

    ClusterPIRResult queryCluster(
        int clusterId,
        const uint8_t* query,
        PrivateHNSWServerV2& server,
        PrivateSearchStatsV2& stats
    );

    int privateGreedySearch(
        const uint8_t* query,
        int entryPoint,
        int targetLevel
    );

    bool isClusterCached(int clusterId) const;
    int64_t getCachedDistance(int newId) const;
    bool hasNeighborInCluster(int newId, int clusterId) const;

    struct SubgroupCiphertextCache {
        std::shared_ptr<NbrMatrix> answer;
        std::shared_ptr<NbrMatrix> Hs;
        std::vector<bool> decryptedNodes;
        int subgroupSize;
    };
    std::unordered_map<int, SubgroupCiphertextCache> subgroupCiphertextCache_;

    void querySubgroupCiphertext(
        int subgroupColumn,
        PrivateHNSWServerV2& server,
        PrivateSearchStatsV2& stats
    );

    void batchQuerySubgroups(
        const std::vector<int>& subgroupColumns,
        PrivateHNSWServerV2& server,
        PrivateSearchStatsV2& stats
    );

    void decryptNodeNeighbors(
        int newId,
        int subgroupColumn,
        int localIdx,
        PrivateSearchStatsV2& stats
    );

    void clearSubgroupCiphertextCache();

    PrivateHNSWMetadataV2 metadata_;
    PrivateHNSWConfigV2 config_;

    simplepir::EmbeddingPIRParams embPirParams_;
    EmbPIRClient embPirClient_;

    simplepir::NeighborPIRParams nbrPirParams_;
    NbrPIRClient nbrPirClient_;

    std::vector<EmbPIRPrecompute> embPool_;
    size_t embPoolIdx_ = 0;

    std::vector<NbrPIRPrecompute> nbrPool_;
    size_t nbrPoolIdx_ = 0;

    std::vector<std::shared_ptr<EmbMatrix>> embQueryPool_;
    std::vector<std::shared_ptr<NbrMatrix>> nbrQueryPool_;
    size_t embQueryPoolIdx_ = 0;
    size_t nbrQueryPoolIdx_ = 0;

    std::shared_ptr<EmbMatrix> getEmbQueryBuffer();
    std::shared_ptr<NbrMatrix> getNbrQueryBuffer();

    EmbPIRPrecompute getEmbPrecompute();
    NbrPIRPrecompute getNbrPrecompute();

    int estimatePoolSize(int ef) const;

    int lastEmbPoolUsed_ = 0;
    int lastNbrPoolUsed_ = 0;
    double avgEmbPoolUsage_ = 0.0;
    double avgNbrPoolUsage_ = 0.0;
    int queryCount_ = 0;

    std::vector<int64_t> innerProductsBuf_;

    std::vector<int64_t> distanceCache_;
    std::vector<std::vector<bool>> neighborCache_;
    std::vector<bool> cachedClusters_;

    std::unordered_map<int, std::vector<int>> neighborNodeIds_;

    struct ClusterCiphertext {
        std::shared_ptr<EmbMatrix> answer;
        std::shared_ptr<EmbMatrix> Hs;
        std::vector<bool> decryptedSubgroups;
        std::vector<int64_t> decryptedDistances;
        bool hasNeighborInfo = false;
        std::vector<std::vector<bool>> neighborClusters;
    };
    std::unordered_map<int, ClusterCiphertext> clusterCiphertextCache_;

    mutable int totalRowsDecrypted_ = 0;
    mutable int totalRowsSkipped_ = 0;

    bool isReady_ = false;
};

class PrivateHNSWUtilsV2 {
public:
    static std::shared_ptr<EmbMatrix> generateEmbeddingSharedMatrix(
        const simplepir::EmbeddingPIRParams& params
    );

    static std::shared_ptr<NbrMatrix> generateNeighborSharedMatrix(
        const simplepir::NeighborPIRParams& params
    );
};

}

#endif
