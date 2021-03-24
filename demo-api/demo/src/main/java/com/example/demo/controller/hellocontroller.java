package com.example.demo.controller;


import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import javax.servlet.http.HttpServletRequest;


@RestController()
@RequestMapping("/home")
public class hellocontroller {

    @GetMapping("/hello")
    public String hello(HttpServletRequest req)

    {
        String name=req.getParameter("name");
        return  ("Hello " + name);
    }

}

