#pragma once
#include <string>
#include "codes_utilities.hpp"
#include "parameter_sequence.hpp"
#include "sequence_utilities.hpp"

/**
 * \addtogroup Sequences
 * \ingroup FUKA
 * @{*/
using namespace ::Kadath::FUKA_Config;

namespace Kadath::FUKA_Solvers {

template <class... Ts>
std::string get_param_str__idx(BCO_PARAMS param_idx, Ts... BCOidx) {
    auto [param_key, paired_param_idx] =
        get_key_val_pair_from_val(MBCO_PARAMS, param_idx);
    return param_key;
}

template <typename U, typename F, class TupType, size_t... I>
U call_F_using_tuple__helper(F&& func,
                             const TupType& _tup,
                             std::index_sequence<I...>) {
    return func(std::get<I>(_tup)...);
}

template <typename U, typename F, class... t>
U call_F_using_tuple(F&& func, const std::tuple<t...>& _tup) {
    return call_F_using_tuple__helper<U>(
        std::forward<F>(func),
        _tup,
        std::make_index_sequence<sizeof...(t)>());
}

template <class... Ts>
class bco_sequence;

template <class... Us>
std::ostream& operator<<(std::ostream&, const bco_sequence<Us...>&);

template <class... Ts>
class bco_sequence : public Parameter_sequence<Ts...> {
   protected:
    internal_variable_simple(std::tuple<Ts...>, mass_fixing_idx);
    internal_variable(double, mass_fixing_val);

    internal_variable_simple(std::tuple<Ts...>, spin_fixing_idx);
    internal_variable(double, spin_fixing_val);

   public:
    bco_sequence()
        : Parameter_sequence<Ts...>(),
          mass_fixing_val(std::nan("1")),
          spin_fixing_val(std::nan("1")) {}

    /**
   * @brief Construct a new Parameter_sequence object from a string and tuple
   * objects
   *
   * @param _str Input parameter string
   * @param _t Tuple containing Config indicies for the desired parameter
   */
    bco_sequence(std::string _str, std::tuple<Ts...> _t)
        : Parameter_sequence<Ts...>(_str, _t),
          mass_fixing_val(std::nan("1")),
          spin_fixing_val(std::nan("1")) {}

    /**
   * @brief Construct a new Parameter_sequence object from a string and a pack
   * of indices
   *
   * @param _str Input parameter string
   * @param _ts Parameter pack consisting of the desired Config indices
   */

    bco_sequence(std::string _str, Ts... _ts)
        : Parameter_sequence<Ts...>(_str, _ts...),
          mass_fixing_val(std::nan("1")),
          spin_fixing_val(std::nan("1")) {}

    bco_sequence(Parameter_sequence<Ts...> const& seq)
        : Parameter_sequence<Ts...>(seq),
          mass_fixing_val(std::nan("1")),
          spin_fixing_val(std::nan("1")) {}

    bool is_mass_set() const { return !std::isnan(mass_fixing_val); }

    bool is_spin_set() const { return !std::isnan(spin_fixing_val); }

    std::string mass_str() const {
        return call_F_using_tuple<std::string>(
            [](auto first, auto... args) {
                return get_param_str__idx(first, args...);
            },
            this->mass_fixing_idx);
    }

    std::string spin_str() const {
        return call_F_using_tuple<std::string>(
            [](auto first, auto... args) {
                return get_param_str__idx(first, args...);
            },
            this->spin_fixing_idx);
    }

    template <class... Us>
    friend std::ostream& operator<<(std::ostream&, const bco_sequence<Us...>&);
};

template <class... idx_t>
inline bool param_idx_is_mass_fixing(BCO_PARAMS const& idx, idx_t... BCOidx) {
    bool is_mass_fixing{false};
    switch (idx) {
        // BH
        case BCO_PARAMS::MCH:
        case BCO_PARAMS::MIRR:
        // NS
        case BCO_PARAMS::HC:
        case BCO_PARAMS::NC:
        case BCO_PARAMS::MADM:
        case BCO_PARAMS::MB:
            is_mass_fixing = true;
            break;
        default:
            break;
    }
    return is_mass_fixing;
}

template <class... idx_t>
inline bool param_idx_is_spin_fixing(BCO_PARAMS const& idx, idx_t... BCOidx) {
    bool is_spin_fixing{false};
    switch (idx) {
        case BCO_PARAMS::OMEGA:
        case BCO_PARAMS::CHI:
        case BCO_PARAMS::JADM:
            is_spin_fixing = true;
            break;
        default:
            break;
    }
    return is_spin_fixing;
}

template <class... idx_t>
bco_sequence<idx_t...> parse_fixed_tree(Tree const& branch,
                                        std::string const parameter_str,
                                        idx_t... idx) {
    double _fixed = std::nan("1");

    // bool vset = seqset = false;
    extract_seq(branch, parameter_str, _fixed);

    bco_sequence<idx_t...> seq(parameter_str, std::make_tuple(idx...));

    if (param_idx_is_mass_fixing(idx...)) {
        seq.set_mass_fixing_idx() = std::make_tuple(idx...);
        seq.set_mass_fixing_val() = _fixed;
    } else if (param_idx_is_spin_fixing(idx...)) {
        seq.set_spin_fixing_idx() = std::make_tuple(idx...);
        seq.set_spin_fixing_val() = _fixed;
    }

    return seq;
}

template <class map_t, class... idx_t>
decltype(auto) find_fixing_values(Tree const& branch,
                                  map_t const& map,
                                  std::string const default_mass_param,
                                  std::string const default_spin_param,
                                  idx_t... idx) {
    auto res = parse_fixed_tree(branch, "", (*map.begin()).second, idx...);

    for (const auto& [key, index] : map) {
        auto tmp_res = parse_fixed_tree(branch, key + "_fixed", index, idx...);
        if (tmp_res.is_mass_set()) {
            if (res.is_mass_set()) {
                std::string msg{
                    "Mass can only be fixed by one value. Check your "
                    "config!\n"};
                throw std::runtime_error(msg.c_str());
            }
            res.set_mass_fixing_idx() = tmp_res.get_mass_fixing_idx();
            res.set_mass_fixing_val() = tmp_res.get_mass_fixing_val();
        } else if (tmp_res.is_spin_set()) {
            if (res.is_spin_set()) {
                std::string msg{
                    "Spin can only be fixed by one value. Check your "
                    "config!\n"};
                throw std::runtime_error(msg.c_str());
            }
            res.set_spin_fixing_idx() = tmp_res.get_spin_fixing_idx();
            res.set_spin_fixing_val() = tmp_res.get_spin_fixing_val();
        }
    }
    if (!res.is_mass_set()) {
        // Try default mass param
        auto tmp_res = parse_fixed_tree(branch,
                                        default_mass_param,
                                        (*map.find(default_mass_param)).second,
                                        idx...);

        if (tmp_res.is_mass_set()) {
            res.set_mass_fixing_idx() = tmp_res.get_mass_fixing_idx();
            res.set_mass_fixing_val() = tmp_res.get_mass_fixing_val();
        } else {
            std::string msg{
                "No mass fixing value not found. Check your config!\n"};
            throw std::runtime_error(msg.c_str());
        }
    }
    if (!res.is_spin_set()) {
        // Try default spin param
        auto tmp_res = parse_fixed_tree(branch,
                                        default_spin_param,
                                        (*map.find(default_spin_param)).second,
                                        idx...);

        if (tmp_res.is_spin_set()) {
            res.set_spin_fixing_idx() = tmp_res.get_spin_fixing_idx();
            res.set_spin_fixing_val() = tmp_res.get_spin_fixing_val();
        } else {
            std::string msg{
                "No spin fixing value not found. Check your config!\n"};
            throw std::runtime_error(msg.c_str());
        }
    }

    return res;
}

inline decltype(auto) find_fixing_values_binary(Tree const& tree, NODES bco) {
    std::string const branch_name{"binary"};

    auto const& bco_param_map{MBCO_PARAMS};

    Tree branch = read_branch(tree, branch_name);

    auto const& node_map{MBCO};

    std::array<NODES, 2> bcos{NODES::BCO1, NODES::BCO2};
    for (const auto& node : branch) {
        if (!node.second.empty()) {
            std::string const node_str = node.first;
            int const tstr_suffix = std::atoi(&node_str.back());

            if (tstr_suffix != bco + 1)
                continue;

            // remove suffix (e,g, bh1 -> bh, ns1 -> ns)
            auto const tstr = node_str.substr(0, 2);

            const auto& it = node_map.find(tstr);

            auto const default_mass_param = (tstr == "bh") ? "mch" : "madm";
            auto const default_spin_param = "chi";

            Tree const child_branch = read_branch(branch, node_str);

            if (it != node_map.end()) {
                auto res = find_fixing_values(child_branch,
                                              MBCO_PARAMS,
                                              default_mass_param,
                                              default_spin_param,
                                              bco);
                return res;
            }
        }
    }
    return parse_fixed_tree(branch, "", (*bco_param_map.begin()).second, BCO1);
}

template <class... Ts>
std::ostream& operator<<(std::ostream& out, const bco_sequence<Ts...>& params) {
    auto indices = params.get_indices();
    std::string s = params.str() + " fixing parameters";
    std::string title = stdio_header(s);
    out << title << std::endl;
    if (params.is_mass_set()) {
        out << std::setw(20) << "mass parameter"
            << ": " << params.mass_str() << '\n'
            << std::setw(20) << "mass value"
            << ": " << params.get_mass_fixing_val() << '\n'
            << std::setw(20) << "mass indices"
            << ": " << params.get_mass_fixing_idx() << '\n';
    } else {
        out << "No Mass fixing parameters set\n";
    }

    if (params.is_spin_set()) {
        out << std::setw(20) << "spin parameter"
            << ": " << params.spin_str() << '\n'
            << std::setw(20) << "spin value"
            << ": " << params.get_spin_fixing_val() << '\n'
            << std::setw(20) << "spin indices"
            << ": " << params.get_spin_fixing_idx() << '\n';
    } else {
        out << "No Spin fixing parameters set\n";
    }
    out << std::endl;

    s = params.str() + " sequence";
    title = stdio_header(s);
    out << title << std::endl;
    if (params.is_set() || params.is_default_set()) {
        if (params.is_set()) {
            out << std::setw(20) << "initial"
                << ": " << params.init() << '\n'
                << std::setw(20) << "final"
                << ": " << params.final() << '\n'
                << std::setw(20) << "step"
                << ": " << params.step_size() << '\n';
            if (params.str() != "res")
                out << std::setw(20) << "# Sequences"
                    << ": " << params.iterations() << '\n';
        } else if (params.is_default_set()) {
            out << std::setw(20) << "Value"
                << ": " << params.default_val() << '\n';
        }
        out << std::setw(20) << "indices"
            << ": " << indices << '\n';
    } else {
        out << "Empty Sequence\n";
    }
    return out;
}

/** @}*/
}  // namespace Kadath::FUKA_Solvers
