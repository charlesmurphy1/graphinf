#ifndef GRAPHINF_UTIL_MCMC_H
#define GRAPHINF_UTIL_MCMC_H

#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include "BaseGraph/types.h"
#include "GraphInf/types.h"
#include "GraphInf/utility/maps.hpp"

namespace GraphInf
{
    struct GraphMove
    {
        GraphMove(std::vector<BaseGraph::Edge> removedEdges, std::vector<BaseGraph::Edge> addedEdges) : removedEdges(removedEdges), addedEdges(addedEdges) {}
        GraphMove() {}
        std::vector<BaseGraph::Edge> removedEdges;
        std::vector<BaseGraph::Edge> addedEdges;

        std::string alias() const
        {
            if (removedEdges.size() == 0 and addedEdges.size() == 0)
                return "none";
            if (removedEdges.size() == 1 and addedEdges.size() == 0)
                return "removed";
            if (removedEdges.size() == 0 and addedEdges.size() == 1)
                return "added";
            if (removedEdges.size() == 1 and addedEdges.size() == 1)
                return "hinge_flip";
            if (removedEdges.size() == 2 and addedEdges.size() == 2)
                return "double_edge_flip";
            return "multiflip";
        }

        friend std::ostream &operator<<(std::ostream &os, const GraphMove &move)
        {
            os << move.display();
            return os;
        }

        std::string display() const
        {
            std::stringstream ss;
            ss << "GraphMove(removed=[";
            for (auto e : removedEdges)
            {
                ss << " {" << e.first << ", " << e.second << "}, ";
            }
            ss << "], added=[";
            for (auto e : addedEdges)
            {
                ss << "{" << e.first << ", " << e.second << "}, ";
            }
            ss << "])";
            return ss.str();
        }
        bool operator==(const GraphMove &other)
        {
            return other.removedEdges == removedEdges and other.addedEdges == addedEdges;
        }
    };

    struct ParamMove
    {
        ParamMove(std::string key = "none", double value = 0.0) : key(key), value(value) {}

        std::string key;
        double value;

        std::string alias() const
        {
            return "param[" + key + "]";
        }

        friend std::ostream &operator<<(std::ostream &os, const ParamMove &move)
        {
            os << move.display();
            return os;
        }

        std::string display() const
        {
            std::stringstream ss;
            ss << "ParamMove(" << key << "=" << value << ")";
            return ss.str();
        }
    };

    template <typename Label>
    struct LabelMove
    {
        LabelMove(BaseGraph::VertexIndex vertexIndex = 0, Label prevLabel = 0, Label nextLabel = 0, int addedLabels = 0, Level level = 0) : vertexIndex(vertexIndex), prevLabel(prevLabel),
                                                                                                                                            nextLabel(nextLabel), addedLabels(addedLabels),
                                                                                                                                            level(level) {}
        BaseGraph::VertexIndex vertexIndex;
        Label prevLabel;
        Label nextLabel;
        int addedLabels;
        Level level;

        std::string alias() const
        {
            return "label_swap";
        }

        friend std::ostream &operator<<(std::ostream &os, const LabelMove<Label> &move)
        {
            os << move.display();
            return os;
        }

        std::string display() const
        {
            std::stringstream ss;
            ss << "LabelMove(vertex=" << vertexIndex;
            ss << ", prevLabel=" << prevLabel;
            ss << ", nextLabel=" << nextLabel;
            ss << ", addedLabels=" << addedLabels;
            ss << ", level=" << level << ")";
            return ss.str();
        }
        bool operator==(const LabelMove &other)
        {
            return other.vertexIndex == vertexIndex and other.prevLabel == prevLabel and other.nextLabel == nextLabel and other.addedLabels == addedLabels and other.level == level;
        }
    };

    template <typename MoveType>
    struct StepResult
    {
        MoveType move;
        double logJointRatio;
        bool accepted;
    };

    struct MCMCSummary
    {
        std::map<std::string, size_t> total, accepted;
        double logJointRatio;

        MCMCSummary(double logJointRatio = 0.0) : logJointRatio(logJointRatio) {}

        template <typename MoveType>
        void update(const StepResult<MoveType> &step)
        {
            if (step.accepted)
            {
                logJointRatio += step.logJointRatio;
                if (accepted.count(step.move.alias()) == 0)
                    accepted[step.move.alias()] = 1;
                else
                    accepted[step.move.alias()]++;
            }
            if (total.count(step.move.alias()) == 0)
                total[step.move.alias()] = 1;
            else
                total[step.move.alias()]++;
        }
        void join(const MCMCSummary &other)
        {
            logJointRatio += other.logJointRatio;
            for (auto &a : other.accepted)
            {
                if (accepted.count(a.first) == 0)
                    accepted[a.first] = a.second;
                else
                    accepted[a.first] += a.second;
            }
            for (auto &t : other.total)
            {
                if (total.count(t.first) == 0)
                    total[t.first] = t.second;
                else
                    total[t.first] += t.second;
            }
        }
        std::string display() const
        {
            std::stringstream ss;
            ss << "MCMCSummary(log_joint_ratio=" << logJointRatio << ", accepted={";
            for (auto &a : accepted)
            {
                ss << a.first << ": " << a.second << ", ";
            }
            ss << "}, total={";
            for (auto &t : total)
            {
                ss << t.first << ": " << t.second << ", ";
            }
            ss << "})";
            return ss.str();
        }
    };

    using BlockMove = LabelMove<BlockIndex>;
}
#endif