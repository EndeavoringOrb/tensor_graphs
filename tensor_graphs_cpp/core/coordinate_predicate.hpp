#pragma once

#include "core/types.hpp"
#include <functional>

// Immutable integer-coordinate expression tree. Comparisons are affine expressions
// against zero; coordinates are mathematical integers, not byte offsets.
class CoordinatePredicate
{
  public:
    enum class Kind { False, True, Less, LessEqual, Equal, And, Or, Not };

    struct AffineExpression
    {
        std::vector<int64_t> coefficients;
        int64_t offset = 0;
    };

    struct Expression
    {
        Kind kind;
        AffineExpression affine;
        std::vector<std::shared_ptr<const Expression>> children;
    };

  private:
    size_t rank_;
    std::shared_ptr<const Expression> expression_;

    static int64_t checkedAdd(int64_t a, int64_t b)
    {
        if ((b > 0 && a > std::numeric_limits<int64_t>::max() - b) ||
            (b < 0 && a < std::numeric_limits<int64_t>::min() - b))
            throw std::overflow_error("[CoordinatePredicate] integer addition overflow");
        return a + b;
    }

    static int64_t checkedMultiply(int64_t a, int64_t b)
    {
        const auto low = std::numeric_limits<int64_t>::min();
        const auto high = std::numeric_limits<int64_t>::max();
        if (a != 0 && b != 0 &&
            ((a > 0 && b > 0 && a > high / b) ||
             (a > 0 && b < 0 && b < low / a) ||
             (a < 0 && b > 0 && a < low / b) ||
             (a < 0 && b < 0 && a < high / b)))
            throw std::overflow_error("[CoordinatePredicate] integer multiplication overflow");
        return a * b;
    }

    CoordinatePredicate(size_t rank, std::shared_ptr<const Expression> expression)
        : rank_(rank), expression_(std::move(expression)) {}

    CoordinatePredicate combine(const CoordinatePredicate &other, Kind kind) const
    {
        if (rank_ != other.rank_)
            throw std::invalid_argument("[CoordinatePredicate] rank mismatch");
        const Kind identity = kind == Kind::And ? Kind::True : Kind::False;
        const Kind absorbing = kind == Kind::And ? Kind::False : Kind::True;
        if (expression_->kind == absorbing || other.expression_->kind == absorbing)
            return constant(rank_, absorbing == Kind::True);
        if (expression_->kind == identity)
            return other;
        if (other.expression_->kind == identity || expression_ == other.expression_)
            return *this;
        return CoordinatePredicate(rank_, std::make_shared<Expression>(
            Expression{kind, {}, {expression_, other.expression_}}));
    }

  public:
    static CoordinatePredicate constant(size_t rank, bool value)
    {
        return CoordinatePredicate(rank, std::make_shared<Expression>(
            Expression{value ? Kind::True : Kind::False, {}, {}}));
    }

    static CoordinatePredicate compare(AffineExpression affine, Kind kind)
    {
        if (kind != Kind::Less && kind != Kind::LessEqual && kind != Kind::Equal)
            throw std::invalid_argument("[CoordinatePredicate] invalid comparison kind");
        const size_t rank = affine.coefficients.size();
        if (std::all_of(affine.coefficients.begin(), affine.coefficients.end(),
                        [](int64_t coefficient) { return coefficient == 0; }))
            return constant(rank, kind == Kind::Less ? affine.offset < 0 :
                                  kind == Kind::LessEqual ? affine.offset <= 0 : affine.offset == 0);
        return CoordinatePredicate(rank, std::make_shared<Expression>(
            Expression{kind, std::move(affine), {}}));
    }

    static CoordinatePredicate fromRegion(const Region &region)
    {
        const size_t rank = region.region.size();
        auto result = constant(rank, true);
        for (size_t axis = 0; axis < rank; ++axis)
        {
            const auto &dim = region.region[axis];
            if (dim.start >= dim.stop)
                return constant(rank, false);
            AffineExpression lower{std::vector<int64_t>(rank, 0), dim.start};
            lower.coefficients[axis] = -1;
            AffineExpression upper{std::vector<int64_t>(rank, 0), -int64_t(dim.stop)};
            upper.coefficients[axis] = 1;
            result = result.intersect(compare(std::move(lower), Kind::LessEqual))
                           .intersect(compare(std::move(upper), Kind::Less));
        }
        return result;
    }

    static CoordinatePredicate fromRegions(size_t rank, const std::vector<Region> &regions)
    {
        std::vector<CoordinatePredicate> predicates;
        predicates.reserve(regions.size());
        for (const auto &region : regions)
        {
            if (region.region.size() != rank)
                throw std::invalid_argument("[CoordinatePredicate] region rank mismatch");
            predicates.push_back(fromRegion(region));
        }
        // Keep imported row lists balanced rather than constructing an O(n)
        // recursion depth. Analytic predicates such as triangular() remain O(1).
        while (predicates.size() > 1)
        {
            std::vector<CoordinatePredicate> next;
            next.reserve((predicates.size() + 1) / 2);
            for (size_t i = 0; i < predicates.size(); i += 2)
                next.push_back(i + 1 < predicates.size() ? predicates[i].unite(predicates[i + 1]) : predicates[i]);
            predicates = std::move(next);
        }
        return predicates.empty() ? constant(rank, false) : predicates.front();
    }

    static CoordinatePredicate triangular(size_t rank, size_t row_axis, size_t col_axis, int64_t diag_k)
    {
        if (row_axis >= rank || col_axis >= rank || row_axis == col_axis)
            throw std::invalid_argument("[CoordinatePredicate] invalid triangular axes");
        AffineExpression affine{std::vector<int64_t>(rank, 0), checkedMultiply(diag_k, -1)};
        affine.coefficients[col_axis] = 1;
        affine.coefficients[row_axis] = -1;
        return compare(std::move(affine), Kind::Less);
    }

    size_t rank() const { return rank_; }
    const Expression &expression() const { return *expression_; }
    CoordinatePredicate intersect(const CoordinatePredicate &other) const { return combine(other, Kind::And); }
    CoordinatePredicate unite(const CoordinatePredicate &other) const { return combine(other, Kind::Or); }
    CoordinatePredicate subtract(const CoordinatePredicate &other) const { return intersect(other.complement()); }

    // Complement is over the integer lattice. Intersect with the tensor bounds
    // when a complement restricted to a tensor is required.
    CoordinatePredicate complement() const
    {
        if (expression_->kind == Kind::True || expression_->kind == Kind::False)
            return constant(rank_, expression_->kind == Kind::False);
        if (expression_->kind == Kind::Not)
            return CoordinatePredicate(rank_, expression_->children[0]);
        return CoordinatePredicate(rank_, std::make_shared<Expression>(
            Expression{Kind::Not, {}, {expression_}}));
    }

    bool contains(const std::vector<int64_t> &coordinate) const
    {
        if (coordinate.size() != rank_)
            throw std::invalid_argument("[CoordinatePredicate] coordinate rank mismatch");
        std::function<bool(const Expression &)> evaluate = [&](const Expression &node) {
            switch (node.kind)
            {
            case Kind::True: return true;
            case Kind::False: return false;
            case Kind::And: return evaluate(*node.children[0]) && evaluate(*node.children[1]);
            case Kind::Or: return evaluate(*node.children[0]) || evaluate(*node.children[1]);
            case Kind::Not: return !evaluate(*node.children[0]);
            default:
                int64_t value = node.affine.offset;
                for (size_t axis = 0; axis < rank_; ++axis)
                    value = checkedAdd(value, checkedMultiply(node.affine.coefficients[axis], coordinate[axis]));
                return node.kind == Kind::Less ? value < 0 :
                       node.kind == Kind::LessEqual ? value <= 0 : value == 0;
            }
        };
        return evaluate(*expression_);
    }

    // Pull back through a view: old_coordinate[i] = coordinates[i](new_coordinate).
    // Handles permutations, sliced offsets/steps, and insertion of singleton axes
    // without enumerating coordinates. General reshape needs non-affine indexing.
    CoordinatePredicate substitute(size_t new_rank, const std::vector<AffineExpression> &coordinates) const
    {
        if (coordinates.size() != rank_)
            throw std::invalid_argument("[CoordinatePredicate] substitution rank mismatch");
        for (const auto &coordinate : coordinates)
            if (coordinate.coefficients.size() != new_rank)
                throw std::invalid_argument("[CoordinatePredicate] substitution expression rank mismatch");
        std::function<CoordinatePredicate(const Expression &)> transform = [&](const Expression &node) {
            if (node.kind == Kind::True || node.kind == Kind::False)
                return constant(new_rank, node.kind == Kind::True);
            if (node.kind == Kind::Not)
                return transform(*node.children[0]).complement();
            if (node.kind == Kind::And || node.kind == Kind::Or)
                return transform(*node.children[0]).combine(transform(*node.children[1]), node.kind);
            AffineExpression affine{std::vector<int64_t>(new_rank, 0), node.affine.offset};
            for (size_t axis = 0; axis < rank_; ++axis)
            {
                const int64_t coefficient = node.affine.coefficients[axis];
                affine.offset = checkedAdd(affine.offset, checkedMultiply(coefficient, coordinates[axis].offset));
                for (size_t target = 0; target < new_rank; ++target)
                    affine.coefficients[target] = checkedAdd(affine.coefficients[target],
                        checkedMultiply(coefficient, coordinates[axis].coefficients[target]));
            }
            return compare(std::move(affine), node.kind);
        };
        return transform(*expression_);
    }

    // Stable structural serialization, suitable for dumps. This is deliberately
    // not a semantic equivalence claim: different trees can describe the same set.
    std::string toString() const
    {
        std::ostringstream out;
        out << "predicate[" << rank_ << "]";
        std::function<void(const Expression &)> write = [&](const Expression &node) {
            out << '(' << static_cast<int>(node.kind);
            if (node.kind == Kind::Less || node.kind == Kind::LessEqual || node.kind == Kind::Equal)
            {
                out << ':' << node.affine.offset;
                for (int64_t coefficient : node.affine.coefficients)
                    out << ',' << coefficient;
            }
            for (const auto &child : node.children)
                write(*child);
            out << ')';
        };
        write(*expression_);
        return out.str();
    }

    bool operator==(const CoordinatePredicate &other) const
    {
        return rank_ == other.rank_ && (expression_ == other.expression_ || toString() == other.toString());
    }
    bool operator!=(const CoordinatePredicate &other) const { return !(*this == other); }
};
