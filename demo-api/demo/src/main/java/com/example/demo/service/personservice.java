package com.example.demo.service;
import com.example.demo.model.person;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;


@Service
public class personservice {

    public String persondetail(person p)
    {
        return ("Hello " + p.getName() +". Your age is "+ p.getAge()+ ". Your nationality is "+ p.getCountry());
        //return "ab";
    }
}
