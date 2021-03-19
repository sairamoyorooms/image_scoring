package com.example.demo.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import com.example.demo.model.person;
import com.example.demo.service.personservice;
import javax.servlet.http.HttpServletRequest;


@RestController
@RequestMapping("/home")
public class personcontroller {
    person p=new person();
    @Autowired
    public personservice pservice;
     @GetMapping("/test")
    public String xyz()
     {
         return "Hello Person";
     }
     @GetMapping("/person")
    public String person(HttpServletRequest request)
     {
         p.setName((String) request.getParameter("name"));
         String age= request.getParameter("age");
         p.setAge(Integer.parseInt(age));
         p.setCountry((String) request.getParameter("country"));
         return pservice.persondetail(p);
     }
}
