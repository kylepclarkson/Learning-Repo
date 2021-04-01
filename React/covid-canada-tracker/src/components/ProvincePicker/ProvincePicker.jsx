import React, {useState} from 'react'

import { Select } from '@material-ui/core'


const ProvincePicker = (props) => {

    const options = [
        { value: "AB", label: "Alberta" },
        { value: "BC", label: "British Columbia" },
        { value: "MB", label: "Manitoba" },
        { value: "NB", label: "New Brunswick" },
        { value: "NL", label: "NL" },
        { value: "NT", label: "NWT" },
        { value: "NS", label: "Nova Scotia" },
        { value: "NU", label: "Nunavut" },
        { value: "ON", label: "Ontario" },
        { value: "PE", label: "PEI" },
        { value: "QC", label: "Quebec" },
        { value: "SK", label: "Saskatchewan" },
        { value: "YT", label: "Yukon" },
    ]

    const [province, setProvince ] = useState({value: 'canada', label: 'Canada'});
    
    // set province
    const [provinceSelect] = useState(() => {
        return () => {
            setProvince(province)
        }
    })

    return (
        
        <Select
            value={province}
            onChange={provinceSelect}>
        </Select>
    )
}

export default ProvincePicker;