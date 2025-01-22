export type Point = [number, number];

export interface LayerData {
  experiment_name: string;
  model_config: string;
  layer: number;
  display_labels: string[];
  states_pca: Point[];
  explained_variance: number[];
}

export interface PCARawData {
  [layer: number]: LayerData;
}
