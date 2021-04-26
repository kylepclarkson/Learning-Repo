import React, { Component } from 'react'
import { makeStyles } from '@material-ui/core/styles';
import InputLabel from '@material-ui/core/InputLabel';
import MenuItem from '@material-ui/core/MenuItem';
import FormControl from '@material-ui/core/FormControl';
import Select from '@material-ui/core/Select';

import styles from './RegionPicker.module.css'

const useStyles = makeStyles((theme) => ({
  formControl: {
    margin: theme.spacing(1),
    minWidth: 250,
  },
  selectEmpty: {
    marginTop: theme.spacing(2),
  },
}));

export default function RegionPicker(props) {

  const classes = useStyles();

  const handleChange = (event) => {
    props.onRegionSelect(event.target.value);
  };

  return (
    <div className={styles.container}>
      <FormControl className={classes.formControl}>
      <InputLabel id="demo-simple-select-label">Region</InputLabel>
      <Select
        labelId="demo-simple-select-label"
        id="demo-simple-select"
        defaultValue={"canada"}
        onChange={handleChange}
      >
        <MenuItem value="canada">Canada</MenuItem>
        <MenuItem value="AB">Alberta</MenuItem>
        <MenuItem value="BC">British Columbia</MenuItem>
        <MenuItem value="MB">Manitoba</MenuItem>
        <MenuItem value="NB">New Brunswick</MenuItem>
        <MenuItem value="NL">Newfoundland and Labrador</MenuItem>
        <MenuItem value="NT">Northwest Territories</MenuItem>
        <MenuItem value="NS">Nova Scotia</MenuItem>
        <MenuItem value="NU">Nunavut</MenuItem>
        <MenuItem value="ON">Ontario</MenuItem>
        <MenuItem value="PE">Prince Edward Island</MenuItem>
        <MenuItem value="QC">Quebec</MenuItem>
        <MenuItem value="SK">Saskatchewan</MenuItem>
        <MenuItem value="YT">Yukon</MenuItem>
        <MenuItem value="RP">Repatriated</MenuItem>
      </Select>
    </FormControl>
    </div>
  )
}

{/* <select onChange={this.handleSelect} defaultValue="canda">
  <option value="canada">Canada</option>
  <option value="AB">Alberta</option>
  <option value="BC">British Columbia</option>
  <option value="MB">Manitoba</option>
  <option value="NB">New Brunswick</option>
  <option value="NL">Newfoundland and Labrador</option>
  <option value="NT">Northwest Territories</option>
  <option value="NS">Nova Scotia</option>
  <option value="NU">Nunavut</option>
  <option value="ON">Ontario</option>
  <option value="PE">Prince Edward Island</option>
  <option value="QC">Quebec</option>
  <option value="SK">Saskatchewan</option>
  <option value="YT">Yukon</option>
  <option value="RP">Repatriated</option>
</select> */}