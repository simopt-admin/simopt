export type Page = "Simulator" | "User Guide" | "About Us";
export type SummaryKind = "solver" | "problem" | "plot";
export type EditMode = { kind: SummaryKind; index: number };

export type Param = {
  name: string;
  description: string;
  default: unknown;
  value: string;
};

export type SummaryEntry = {
  name: string;
  params: Param[];
  expanded: boolean;
};

export type PlotSummaryEntry = SummaryEntry & {
  solvers: string[];
  problems: string[];
};

export type SchemaParam = {
  name: string;
  label: string;
  type: "bool" | "int" | "float" | "text" | string;
  default: string | number | boolean | null;
};

export type FixedSchema = { params: SchemaParam[] };
export type FormValue = string | number | boolean | null | undefined;
export type FormValues = Record<string, FormValue>;

export type CompatibilityCell = {
  compatible: boolean;
  message?: string;
};

export type Compatibility = Record<string, Record<string, CompatibilityCell>>;
export type ParamsResponse = { parameters?: Array<Partial<Param>> };
