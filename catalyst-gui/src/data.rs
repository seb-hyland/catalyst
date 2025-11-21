use arrow::{
    array::{Array, BooleanArray, RecordBatch, StructArray},
    compute::{concat_batches, filter},
    csv,
    datatypes::Schema,
    error::ArrowError,
    ffi::{FFI_ArrowArray, FFI_ArrowSchema, from_ffi, to_ffi},
    json,
    util::display::{ArrayFormatter, FormatOptions},
};
use egui::Ui;
use std::{
    collections::HashSet,
    fs::File,
    io::{self, BufReader, Read},
    path::Path,
    sync::Arc,
};

#[derive(Debug)]
pub enum DataError {
    FileRead(io::Error),
    UnknownType(Option<String>),
    SchemaInference(ArrowError),
    Reader(ArrowError),
    BatchConcat(ArrowError),
    Batch(ArrowError),
    Reconstruct(ArrowError),
    Format(ArrowError),
    Ffi(ArrowError),
}

pub enum InputType {
    Csv,
    Json,
}

pub enum InputBuilder {
    Csv(csv::reader::ReaderBuilder),
    Json(json::reader::ReaderBuilder),
}
impl InputBuilder {
    fn build(
        self,
        reader: BufReader<File>,
        schema: Arc<Schema>,
    ) -> Result<Vec<RecordBatch>, DataError> {
        match self {
            Self::Csv(builder) => {
                let reader = builder
                    .build_buffered(reader)
                    .map_err(|e| DataError::Reader(e))?;
                reader
                    .into_iter()
                    .map(|res| res.map_err(|e| DataError::Reader(e)))
                    .collect()
            }
            Self::Json(builder) => {
                let reader = builder.build(reader).map_err(|e| DataError::Reader(e))?;
                reader
                    .into_iter()
                    .map(|res| res.map_err(|e| DataError::Reader(e)))
                    .collect()
            }
        }
    }
}

pub struct ReadFile<'a>(&'a Path);
impl<'a> ReadFile<'a> {
    pub fn next(self, reader: InputBuilder) -> Result<IdentifySchema, DataError> {
        let input_type = match self.0.extension().and_then(|p| p.to_str()) {
            Some("csv") => InputType::Csv,
            Some("json") => InputType::Json,
            other => {
                return Err(DataError::UnknownType(
                    other.map(<str as ToOwned>::to_owned),
                ));
            }
        };

        let open_file = || File::open(self.0).map_err(|e| DataError::FileRead(e));
        let schema_reader = BufReader::new(open_file()?);
        let primary_reader = BufReader::new(open_file()?);

        Ok(IdentifySchema {
            schema_reader,
            primary_reader,
            input_type,
        })
    }
}

pub struct IdentifySchema {
    schema_reader: BufReader<File>,
    primary_reader: BufReader<File>,
    input_type: InputType,
}
impl IdentifySchema {
    pub fn next(self) -> Result<DefiningFormat, DataError> {
        let arg1 = self.schema_reader;
        let arg2 = Some(100);

        let (schema_raw, _) = match self.input_type {
            InputType::Csv => csv::reader::Format::default().infer_schema(arg1, arg2),
            InputType::Json => json::reader::infer_json_schema(arg1, arg2),
        }
        .map_err(|e| DataError::SchemaInference(e))?;
        let schema = Arc::new(schema_raw);

        let primary_reader = self.primary_reader;
        let input_type = self.input_type;
        Ok(DefiningFormat {
            primary_reader,
            schema,
            input_type,
        })
    }
}

pub struct DefiningFormat {
    primary_reader: BufReader<File>,
    schema: Arc<Schema>,
    input_type: InputType,
}
impl DefiningFormat {
    pub fn next(self, builder: InputBuilder) -> Result<DataView, DataError> {
        let record_batches = builder.build(self.primary_reader, Arc::clone(&self.schema))?;
        let data = concat_batches(&self.schema, record_batches.iter())
            .map_err(|e| DataError::BatchConcat(e))?;
        Ok(DataView(data))
    }
}

pub struct DataView(RecordBatch);
impl DataView {
    pub fn render(&self, ui: &mut Ui) -> Result<(), DataError> {
        let data = &self.0;
        let schema = data.schema();
        for (col, field) in data.columns().iter().zip(schema.fields.iter()) {
            ui.vertical(|ui| {
                // Column name
                ui.heading(field.name());

                let options = FormatOptions::new().with_null("NULL");
                let formatter = ArrayFormatter::try_new(col, &options)
                    .map_err(|e| DataError::Format(e))
                    .expect("Create popup!");

                for row_idx in 0..col.len() {
                    ui.label(formatter.value(row_idx).to_string());
                }
            });
        }

        Ok(())
    }

    pub fn remove(
        &self,
        cols_to_remove: &HashSet<usize>,
        rows_to_remove: &[usize],
    ) -> Result<Self, DataError> {
        let mut data = self.0.clone();

     // Remove columns
        if !cols_to_remove.is_empty() {
            let schema = data.schema();
            let fields = schema.fields.iter();

            let (filtered_cols, filtered_fields): (Vec<_>, Vec<_>) = data
                .columns()
                .iter()
                .zip(fields)
                .enumerate()
                .filter_map(|(i, (col, field))| {
                    if cols_to_remove.contains(&i) {
                        None
                    } else {
                        Some((Arc::clone(col), Arc::clone(field)))
                    }
                })
                .collect();

            let new_schema = Arc::new(Schema::new(filtered_fields));
            data = RecordBatch::try_new(new_schema, filtered_cols)
                .map_err(|e| DataError::Reconstruct(e))?;
        }

        // Remove rows
        if !rows_to_remove.is_empty() {
            let n_rows = data.num_rows();

            let mut keep_rows = vec![true; n_rows];
            for row in rows_to_remove {
                keep_rows[*row] = false;
            }
            let keep_rows = BooleanArray::from(keep_rows);

            let filtered_data = data
                .columns()
                .iter()
                .map(|col| filter(col, &keep_rows).map_err(|e| DataError::Reconstruct(e)))
                .collect::<Result<Vec<_>, _>>()?;
            data = RecordBatch::try_new(data.schema(), filtered_data)
                .map_err(|e| DataError::Reconstruct(e))?;
        }

        Ok(Self(data))
    }

    pub fn to_ffi(&self) -> Result<Ffi, DataError> {
        let data = &self.0;
        let array = StructArray::from(data.clone());

        let (ffi_data, ffi_schema) = to_ffi(&array.into_data()).map_err(|e| DataError::Ffi(e))?;
        Ok(Ffi(ffi_data, ffi_schema))
    }
}

pub struct Ffi(FFI_ArrowArray, FFI_ArrowSchema);
impl Ffi {
    pub fn to_data_view(self) -> Result<DataView, DataError> {
        let Self(ffi_array, ffi_schema) = self;
        let array_data =
            unsafe { from_ffi(ffi_array, &ffi_schema) }.map_err(|e| DataError::Ffi(e))?;

        let array = StructArray::from(array_data);
        Ok(DataView(RecordBatch::from(array)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::{
        array::record_batch,
        // Fixes internal arrow macro bug
        datatypes as arrow_schema,
    };

    const CSV_DATA: &str = r#"col0,col1,col2,col3,col4
11,12,13,14,15
21,22,23,24,25
31,32,33,34,35
41,42,43,44,45
51,52,53,54,55
"#;

    fn create_readers() -> (BufReader<&'static [u8]>, BufReader<&'static [u8]>) {
        fn inner() -> BufReader<&'static [u8]> {
            BufReader::new(CSV_DATA.as_bytes())
        }
        (inner(), inner())
    }

    //     #[test]
    //     fn test_parse_csv() {
    //         let (reader1, reader2) = create_readers();

    //         let parsed = read_csv(
    //             reader1,
    //             reader2,
    //             csv::reader::Format::default().with_header(true),
    //         )
    //         .expect("Failed to read CSV!");
    //         let expected = record_batch!(
    //             ("col0", Int64, [11, 21, 31, 41, 51]),
    //             ("col1", Int64, [12, 22, 32, 42, 52]),
    //             ("col2", Int64, [13, 23, 33, 43, 53]),
    //             ("col3", Int64, [14, 24, 34, 44, 54]),
    //             ("col4", Int64, [15, 25, 35, 45, 55])
    //         )
    //         .expect("Failed to construct expected `RecordBatch`");

    //         assert_eq!(parsed, expected);
    //     }

    //     #[test]
    //     fn test_remove_components() {
    //         let (reader1, reader2) = create_readers();
    //         let parsed = read_csv(
    //             reader1,
    //             reader2,
    //             csv::reader::Format::default().with_header(true),
    //         )
    //         .expect("Failed to read CSV!");

    //         let cols_to_remove = {
    //             let mut set = HashSet::new();
    //             set.insert(1);
    //             set.insert(3);
    //             set
    //         };
    //         let rows_to_remove = [2, 4];

    //         let filtered =
    //             filter_data(parsed, &cols_to_remove, &rows_to_remove).expect("Failed to filter Arrow");
    //         let expected = record_batch!(
    //             ("col0", Int64, [11, 21, 41]),
    //             ("col2", Int64, [13, 23, 43]),
    //             ("col4", Int64, [15, 25, 45])
    //         )
    //         .expect("Failed to construct expected `RecordBatch`");

    //         assert_eq!(filtered, expected);
    //     }
}
